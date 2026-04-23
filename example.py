"""
Examples showing ProvenanceCapability in single-agent and multi-agent scenarios.

Azure OpenAI (reads AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY). A `.env` file next to
this script is loaded automatically when you run `uv run python example.py`.

    uv sync
    # .env or inline:
    AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/ \\
      AZURE_OPENAI_API_KEY=... \\
      uv run python example.py

Optional: AZURE_OPENAI_DEPLOYMENT (defaults to gpt-4o), OPENAI_API_VERSION for classic
Azure OpenAI endpoints that require an API version query parameter.

Anthropic (alternative):

    ANTHROPIC_API_KEY=... uv run python example.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.run import AgentRunResult

from pydantic_ai_provenance import (
    ProvenanceCapability,
    ProvenanceStore,
    attribute_output,
    to_mermaid,
    verify_citations_sync,
)


def _use_azure_openai() -> bool:
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"))


def _azure_chat_model() -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.azure import AzureProvider

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    return OpenAIChatModel(deployment, provider=AzureProvider())


def _anthropic_model(strong: bool) -> Any:
    from pydantic_ai.models.anthropic import AnthropicModel

    if strong:
        return AnthropicModel("claude-sonnet-4-6")
    return AnthropicModel("claude-haiku-4-5-20251001")


def _example_model(*, strong: bool) -> Any:
    if _use_azure_openai():
        return _azure_chat_model()
    return _anthropic_model(strong)


def _user_prompt_text(content: str | object) -> str:
    if isinstance(content, str):
        return content
    return repr(content)


def print_model_io(result: AgentRunResult[Any], *, heading: str) -> None:
    """Print each model request/response from this run (from `result.new_messages()`)."""
    print("\n" + "=" * 60)
    print(heading)
    print("=" * 60)

    for i, msg in enumerate(result.new_messages(), 1):
        if isinstance(msg, ModelRequest):
            print(f"\n--- [{i}] Model input (request to the LLM) ---")
            if msg.instructions:
                print("(instructions merged into this request)\n")
                print(msg.instructions)
                print()
            for j, part in enumerate(msg.parts, 1):
                if isinstance(part, SystemPromptPart):
                    print(f"[{i}.{j}] system prompt\n{part.content}\n")
                elif isinstance(part, UserPromptPart):
                    print(f"[{i}.{j}] user prompt\n{_user_prompt_text(part.content)}\n")
                elif isinstance(part, ToolReturnPart | BuiltinToolReturnPart):
                    body = part.model_response_str()
                    print(
                        f"[{i}.{j}] tool result: {part.tool_name!r} "
                        f"(tool_call_id={part.tool_call_id!r})\n{body or repr(part.content)}\n"
                    )
                elif isinstance(part, RetryPromptPart):
                    print(f"[{i}.{j}] retry prompt (tool={part.tool_name!r})\n{part.content!r}\n")
                else:
                    print(f"[{i}.{j}] {type(part).__name__}\n{part!r}\n")
        elif isinstance(msg, ModelResponse):
            u = msg.usage
            print(
                f"\n--- [{i}] Model output (response from the LLM; "
                f"model={msg.model_name!r}; tokens in={u.request_tokens} out={u.response_tokens}) ---"
            )
            for j, part in enumerate(msg.parts, 1):
                if isinstance(part, TextPart):
                    print(f"[{i}.{j}] assistant text\n{part.content}\n")
                elif isinstance(part, ToolCallPart | BuiltinToolCallPart):
                    print(f"[{i}.{j}] tool call: {part.tool_name!r}\n{part.args_as_json_str()}\n")
                elif isinstance(part, ThinkingPart) and part.content:
                    clipped = part.content if len(part.content) <= 4000 else part.content[:4000] + "\n…"
                    print(f"[{i}.{j}] thinking ({len(part.content)} chars)\n{clipped}\n")
                else:
                    print(f"[{i}.{j}] {type(part).__name__}\n{part!r}\n")
        else:
            print(f"\n--- [{i}] {type(msg).__name__} ---\n{msg!r}\n")


def print_citation_verification(store: ProvenanceStore, *, label: str, text: str) -> None:
    """Run ``verify_citations_sync`` and print a short report (Steps 1–2)."""
    rep = verify_citations_sync(text, store)
    print("\n" + "-" * 60)
    print(f"Citation verification — {label}")
    print("-" * 60)
    if rep.sanitize_records:
        print(f"Step 1: {len(rep.sanitize_records)} tag(s) adjusted or removed")
        for r in rep.sanitize_records:
            print(f"  {r.raw_tag!r}: removed {r.keys_removed!r}, kept {r.keys_kept!r}")
    else:
        print("Step 1: no invalid keys in citation tags")
    if rep.sanitized_text != text:
        clip = rep.sanitized_text if len(rep.sanitized_text) <= 400 else rep.sanitized_text[:400] + "…"
        print(f"Sanitized text:\n{clip}\n")
    print(f"Step 2: {len(rep.lexical)} lexical alignment row(s)")
    for row in rep.lexical:
        print(
            f"  {row.citation_key!r} (vs {row.source_key!r}) "
            f"overlap={row.overlap_ratio:.3f} claim={row.claim_excerpt!r}…"
        )


# ---------------------------------------------------------------------------
# Example 1: Single agent with a source tool and a compute tool
# ---------------------------------------------------------------------------


async def example_single_agent() -> None:
    print("=" * 60)
    print("Example 1: Single agent")
    print("=" * 60)

    provenance = ProvenanceCapability(
        agent_name="summariser",
        source_tools=["read_file"],  # read_file produces raw-source nodes
    )

    agent = Agent(
        _example_model(strong=True),
        capabilities=[provenance],
        system_prompt="You summarise the content of files.",
    )

    @agent.tool_plain
    def read_file(path: str) -> str:
        """Read the content of a file."""
        return "The quick brown fox jumps over the lazy dog."

    @agent.tool_plain
    def word_count(text: str) -> int:
        """Count the words in a text."""
        return len(text.split())

    result = await agent.run("Read report.txt and tell me the word count.")

    print_model_io(result, heading="Example 1 — model input / output (this run)")

    store = provenance.store
    attribution = attribute_output(store)

    print(f"\nOutput: {result.output}\n")
    print(attribution.summary())
    print("\n--- Mermaid diagram ---")
    print(to_mermaid(store))

    out = str(result.output)
    print_citation_verification(store, label="model final output", text=out)
    summary_keys = list(store.citation_summary().keys())
    f_keys = [k for k in summary_keys if k.startswith("f_")]
    if f_keys:
        demo = (
            f"The file says a quick brown fox jumps. [REF|{f_keys[0]}] "
            "Mixed with [REF|totally_fake_key]."
        )
        print_citation_verification(store, label="synthetic (valid + bogus keys)", text=demo)


# ---------------------------------------------------------------------------
# Example 2: Multi-agent — parent delegates to a research subagent
# ---------------------------------------------------------------------------


async def example_multi_agent() -> None:
    print("\n" + "=" * 60)
    print("Example 2: Multi-agent")
    print("=" * 60)

    research_provenance = ProvenanceCapability(
        agent_name="researcher",
        source_tools=["fetch_url"],
    )
    coordinator_provenance = ProvenanceCapability(
        agent_name="coordinator",
    )

    research_agent = Agent(
        _example_model(strong=False),
        capabilities=[research_provenance],
        system_prompt="You are a research agent. Fetch URLs and summarise their content.",
    )

    import random

    @research_agent.tool_plain
    def fetch_url(url: str) -> str:
        """Fetch content from a URL."""
        facts = [
            "Pydantic AI integrates seamlessly with Python's data modeling tools.",
            "Pydantic AI enables structured agent workflows with provenance tracking.",
            "You can use Pydantic AI to build composable multi-agent systems.",
            "Pydantic AI builds upon Pydantic for powerful data validation.",
            "With Pydantic AI, each agent interaction is traceable and auditable.",
            "Pydantic AI supports citation graphs for fact attribution.",
            "The framework is designed for building robust LLM agent pipelines.",
            "Pydantic AI supports both synchronous and asynchronous agent calls.",
            "It is easy to register tools and capabilities in Pydantic AI agents.",
            "Pydantic AI provides utilities for citation verification and reporting."
        ]
        return random.choice(facts)
   

    coordinator_agent = Agent(
        _example_model(strong=True),
        capabilities=[coordinator_provenance],
        system_prompt="You coordinate research tasks.",
    )

    @coordinator_agent.tool
    async def delegate_research(ctx, topic: str) -> str:  # type: ignore[misc]
        """Delegate a research task to the research subagent."""
        result = await research_agent.run(
            f"Research this topic and summarise: {topic}",
            usage=ctx.usage,
        )
        topic_label = f"{topic[:60]}…" if len(topic) > 60 else topic
        print_model_io(
            result,
            heading=f"Example 2 — research subagent model I/O (topic: {topic_label})",
        )
        return result.output

    result = await coordinator_agent.run(
        "Find out what Pydantic AI is and give me a detailed summary."
    )

    print_model_io(result, heading="Example 2 — coordinator model I/O (this run)")

    # Both agents share the same store (via ContextVar propagation).
    store = coordinator_provenance.store

    print(f"\nOutput: {result.output}\n")
    out = str(result.output)
    print_citation_verification(store, label="coordinator final output", text=out)
    u_keys = [k for k in store.citation_summary().keys() if k.startswith("u_")]
    if u_keys:
        demo = f"Pydantic AI is a framework. [REF|{u_keys[0]}] See [REF|bad_u]."
        print_citation_verification(store, label="synthetic (valid + bogus keys)", text=demo)


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parent / ".env")
    if not _use_azure_openai() and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Configure credentials: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
            "for Azure OpenAI, or ANTHROPIC_API_KEY for Anthropic.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    # sasyncio.run(example_single_agent())
    asyncio.run(example_multi_agent())
