"""
Single-agent provenance example.

One agent with a source tool (``read_file``) and a compute tool (``word_count``).
``ProvenanceCapability`` automatically tracks which file content ends up in the
final answer and annotates it with inline ``[REF|…]`` citation tags.

    uv sync
    ANTHROPIC_API_KEY=... uv run python examples/single_agent.py
    # or Azure OpenAI:
    AZURE_OPENAI_ENDPOINT=https://... AZURE_OPENAI_API_KEY=... uv run python examples/single_agent.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from _common import example_model, print_citation_verification, print_model_io, require_credentials
from dotenv import load_dotenv
from pydantic_ai import Agent

from pydantic_ai_provenance.attribution import attribute_output
from pydantic_ai_provenance.capability import ProvenanceCapability


async def main() -> None:
    print("=" * 60)
    print("Single-agent provenance example")
    print("=" * 60)

    provenance = ProvenanceCapability(
        agent_name="summariser",
        source_tools=["read_file"],
    )

    agent = Agent(
        example_model(strong=True),
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

    print_model_io(result, heading="Single-agent — model input / output")

    store = provenance.store
    attribution = attribute_output(store)

    print(f"\nOutput: {result.output}\n")
    print(attribution.summary())
    print("\n--- Mermaid diagram ---")
    print(store.to_mermaid())

    out = str(result.output)
    await print_citation_verification(store, label="model final output", text=out)

    d_keys = [k for k in store.citation_summary() if k.startswith("d_")]
    if d_keys:
        demo = f"The file says a quick brown fox jumps. [REF|{d_keys[0]}] Mixed with [REF|totally_fake_key]."
        await print_citation_verification(store, label="synthetic (valid + bogus keys)", text=demo)


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    require_credentials()
    asyncio.run(main())
