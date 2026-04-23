"""
Multi-agent provenance example.

A coordinator agent delegates research tasks to a subagent. Both share the same
``ProvenanceStore`` (via ``ContextVar`` propagation), so citation keys from the
subagent's source tools surface in the coordinator's final answer.

After the run the full provenance graph is saved to ``multi_agent_provenance.html``
and opened in the default browser as an interactive Cytoscape.js visualisation.

    uv sync
    ANTHROPIC_API_KEY=... uv run python examples/multi_agent.py
    # or Azure OpenAI:
    AZURE_OPENAI_ENDPOINT=https://... AZURE_OPENAI_API_KEY=... uv run python examples/multi_agent.py
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

from _common import example_model, print_citation_verification, print_model_io, require_credentials
from dotenv import load_dotenv
from pydantic_ai import Agent

from pydantic_ai_provenance.capability import ProvenanceCapability


async def main() -> None:
    print("=" * 60)
    print("Multi-agent provenance example")
    print("=" * 60)

    research_provenance = ProvenanceCapability(
        agent_name="researcher",
        source_tools=["fetch_url"],
    )
    coordinator_provenance = ProvenanceCapability(agent_name="coordinator")

    research_agent = Agent(
        example_model(strong=False),
        capabilities=[research_provenance],
        system_prompt="You are a research agent. Fetch URLs and summarise their content.",
    )

    _FACTS = [
        "Pydantic AI integrates seamlessly with Python's data modeling tools.",
        "Pydantic AI enables structured agent workflows with provenance tracking.",
        "You can use Pydantic AI to build composable multi-agent systems.",
        "Pydantic AI builds upon Pydantic for powerful data validation.",
        "With Pydantic AI, each agent interaction is traceable and auditable.",
        "Pydantic AI supports citation graphs for fact attribution.",
        "The framework is designed for building robust LLM agent pipelines.",
        "Pydantic AI supports both synchronous and asynchronous agent calls.",
        "It is easy to register tools and capabilities in Pydantic AI agents.",
        "Pydantic AI provides utilities for citation verification and reporting.",
    ]

    @research_agent.tool_plain
    def fetch_url(url: str) -> str:
        """Fetch content from a URL."""
        return random.choice(_FACTS)

    coordinator_agent = Agent(
        example_model(strong=True),
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
        label = f"{topic[:60]}…" if len(topic) > 60 else topic
        print_model_io(result, heading=f"Research subagent — model I/O (topic: {label})")
        return result.output

    result = await coordinator_agent.run("Find out what Pydantic AI is and give me a detailed summary.")

    print_model_io(result, heading="Coordinator — model I/O")

    # Both agents share the same store via ContextVar propagation.
    store = coordinator_provenance.store

    print(f"\nOutput: {result.output}\n")
    out = str(result.output)
    await print_citation_verification(store, label="coordinator final output", text=out)

    d_keys = [k for k in store.citation_summary() if k.startswith("d_")]
    if d_keys:
        demo = f"Pydantic AI is a framework. [REF|{d_keys[0]}] See [REF|bad_u]."
        await print_citation_verification(store, label="synthetic (valid + bogus keys)", text=demo)

    # Interactive provenance graph
    html_path = Path(__file__).parent / "multi_agent_provenance.html"
    html_path.write_text(store.to_html(title="Multi-Agent Provenance Graph"), encoding="utf-8")
    print(f"\nProvenance graph saved → {html_path}")
    if "--no-browser" not in sys.argv:
        print("Opening interactive graph in browser…")
        store.open_in_browser(title="Multi-Agent Provenance Graph")


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    require_credentials()
    asyncio.run(main())
