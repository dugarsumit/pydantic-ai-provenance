"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from pydantic_ai_provenance.graph import NodeType, ProvenanceNode
from pydantic_ai_provenance.store import ProvenanceStore


def make_node(
    node_type: NodeType,
    label: str = "test",
    agent_name: str = "agent",
    run_id: str = "run-1",
    **data,
) -> ProvenanceNode:
    return ProvenanceNode.create(
        type=node_type,
        label=label,
        agent_name=agent_name,
        run_id=run_id,
        **data,
    )


@pytest.fixture()
def empty_store() -> ProvenanceStore:
    return ProvenanceStore()


@pytest.fixture()
def simple_store() -> tuple[ProvenanceStore, str, str]:
    """Store with one DATA_READ → TOOL_RESULT chain. Returns (store, data_node_id, citation_key)."""
    store = ProvenanceStore()
    data_node = make_node(NodeType.DATA_READ, "read_file", file_path="report.txt")
    store.add_node(data_node)
    citation_key = store.register_data_source(data_node.id)

    result_node = make_node(
        NodeType.TOOL_RESULT,
        "Result: read_file",
        tool_name="read_file",
        result="The quick brown fox jumps over the lazy dog.",
    )
    store.add_node(result_node)
    store.add_edge(data_node.id, result_node.id, "returns")

    return store, data_node.id, citation_key
