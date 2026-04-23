"""Tests for pydantic_ai_provenance.viz."""

from __future__ import annotations

import json

import pytest

from pydantic_ai_provenance.graph import NodeType, ProvenanceNode
from pydantic_ai_provenance.store import ProvenanceStore
from pydantic_ai_provenance.viz import to_dot, to_json, to_json_str, to_mermaid


def _make_node(node_type: NodeType, label: str) -> ProvenanceNode:
    return ProvenanceNode.create(
        type=node_type, label=label, agent_name="agent", run_id="r"
    )


def _two_node_store() -> ProvenanceStore:
    store = ProvenanceStore()
    src = _make_node(NodeType.DATA_READ, "read_file")
    out = _make_node(NodeType.FINAL_OUTPUT, "Final answer")
    store.add_node(src)
    store.add_node(out)
    store.add_edge(src.id, out.id, "produces")
    return store


# ---------------------------------------------------------------------------
# to_mermaid
# ---------------------------------------------------------------------------


def test_to_mermaid_starts_with_flowchart():
    store = _two_node_store()
    result = to_mermaid(store)
    assert result.startswith("flowchart LR")


def test_to_mermaid_contains_node_labels():
    store = _two_node_store()
    result = to_mermaid(store)
    assert "read_file" in result
    assert "Final answer" in result


def test_to_mermaid_contains_arrow():
    store = _two_node_store()
    result = to_mermaid(store)
    assert "-->" in result


def test_to_mermaid_contains_classDef():
    store = _two_node_store()
    result = to_mermaid(store)
    assert "classDef" in result


def test_to_mermaid_empty_store():
    store = ProvenanceStore()
    result = to_mermaid(store)
    assert result.startswith("flowchart LR")


# ---------------------------------------------------------------------------
# to_dot
# ---------------------------------------------------------------------------


def test_to_dot_starts_with_digraph():
    store = _two_node_store()
    result = to_dot(store)
    assert result.startswith("digraph provenance")


def test_to_dot_contains_node_labels():
    store = _two_node_store()
    result = to_dot(store)
    assert "read_file" in result
    assert "Final answer" in result


def test_to_dot_contains_arrow():
    store = _two_node_store()
    result = to_dot(store)
    assert "->" in result


def test_to_dot_custom_graph_name():
    store = _two_node_store()
    result = to_dot(store, graph_name="my_graph")
    assert "digraph my_graph" in result


def test_to_dot_ends_with_closing_brace():
    store = _two_node_store()
    result = to_dot(store)
    assert result.strip().endswith("}")


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


def test_to_json_returns_dict():
    store = _two_node_store()
    result = to_json(store)
    assert isinstance(result, dict)


def test_to_json_has_nodes_and_edges_keys():
    store = _two_node_store()
    result = to_json(store)
    assert "nodes" in result
    assert "edges" in result


def test_to_json_nodes_have_required_fields():
    store = _two_node_store()
    result = to_json(store)
    for node in result["nodes"]:
        assert "id" in node
        assert "type" in node
        assert "label" in node
        assert "agent_name" in node
        assert "run_id" in node
        assert "timestamp" in node


def test_to_json_edges_have_required_fields():
    store = _two_node_store()
    result = to_json(store)
    for edge in result["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert "label" in edge


def test_to_json_node_count():
    store = _two_node_store()
    result = to_json(store)
    assert len(result["nodes"]) == 2


def test_to_json_edge_count():
    store = _two_node_store()
    result = to_json(store)
    assert len(result["edges"]) == 1


# ---------------------------------------------------------------------------
# to_json_str
# ---------------------------------------------------------------------------


def test_to_json_str_returns_valid_json():
    store = _two_node_store()
    result = to_json_str(store)
    parsed = json.loads(result)
    assert "nodes" in parsed
    assert "edges" in parsed


def test_to_json_str_respects_indent():
    store = _two_node_store()
    result_2 = to_json_str(store, indent=2)
    result_4 = to_json_str(store, indent=4)
    assert "    " in result_4
    assert result_2 != result_4
