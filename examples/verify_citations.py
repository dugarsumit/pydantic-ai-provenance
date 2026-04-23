"""
Offline citation-verification demo — no LLM credentials required.

Exercises ``strip_unresolvable_citation_keys`` (Step 1) and
``verify_citations`` (Step 2) on a synthetic provenance graph:

    uv run python examples/verify_citations.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from _common import _heading, print_citation_verification
from dotenv import load_dotenv

from pydantic_ai_provenance.graph import NodeType, ProvenanceNode
from pydantic_ai_provenance.store import ProvenanceStore


def _build_store(source_body: str, *, file_path: str = "demo.txt") -> tuple[ProvenanceStore, str]:
    """Build a minimal DATA_READ → TOOL_RESULT store and return its citation key."""
    store = ProvenanceStore()
    run_id = "offline-demo"
    data_read = ProvenanceNode.create(NodeType.DATA_READ, "read_file", "demo", run_id, file_path=file_path)
    store.add_node(data_read)
    citation_key = store.register_data_source(data_read.id)
    tool_result = ProvenanceNode.create(
        NodeType.TOOL_RESULT,
        "Result: read_file",
        "demo",
        run_id,
        tool_name="read_file",
        result=source_body,
    )
    store.add_node(tool_result)
    store.add_edge(data_read.id, tool_result.id, "returns")
    return store, citation_key


async def main() -> None:
    print("=" * 60)
    print("Citation verification — offline smoke test")
    print("=" * 60)

    source = "The quick brown fox jumps over the lazy dog near the river bank."
    store, dkey = _build_store(source)

    _heading(["Case A — claim aligns with source (expect non-empty [REF|…] after refine)"])
    await print_citation_verification(
        store,
        label="aligned claim",
        text=f"The passage mentions a quick brown fox. [REF|{dkey}]",
    )

    _heading(["Case B — mixed valid + bogus keys (Step 1 strips bogus)"])
    await print_citation_verification(
        store,
        label="sanitize + verify",
        text=f"A fox appears in the story. [REF|{dkey}|not_a_real_key]",
    )

    _heading(["Case C — claim unrelated to source (expect weak TF-IDF; tag may be dropped)"])
    await print_citation_verification(
        store,
        label="misaligned claim",
        text=f"Quantum entanglement enables superluminal routers. [REF|{dkey}]",
    )


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    asyncio.run(main())
