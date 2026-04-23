"""Tests for pydantic_ai_provenance.store."""

from __future__ import annotations

import pytest

from pydantic_ai_provenance.graph import NodeType, ProvenanceNode
from pydantic_ai_provenance.store import ProvenanceStore


def _make_node(node_type: NodeType, label: str = "node") -> ProvenanceNode:
    return ProvenanceNode.create(
        type=node_type, label=label, agent_name="agent", run_id="r"
    )


# ---------------------------------------------------------------------------
# Citation registry
# ---------------------------------------------------------------------------


def test_register_data_source_returns_d_prefix():
    store = ProvenanceStore()
    node = _make_node(NodeType.DATA_READ)
    store.add_node(node)
    key = store.register_data_source(node.id)
    assert key.startswith("d_")


def test_register_data_source_increments():
    store = ProvenanceStore()
    n1 = _make_node(NodeType.DATA_READ, "n1")
    n2 = _make_node(NodeType.DATA_READ, "n2")
    store.add_node(n1)
    store.add_node(n2)
    k1 = store.register_data_source(n1.id)
    k2 = store.register_data_source(n2.id)
    assert k1 != k2


def test_register_agent_output_returns_a_prefix():
    store = ProvenanceStore()
    node = _make_node(NodeType.FINAL_OUTPUT)
    store.add_node(node)
    key = store.register_agent_output(node.id)
    assert key.startswith("a_")


def test_register_mixed_counter_is_shared():
    store = ProvenanceStore()
    d_node = _make_node(NodeType.DATA_READ)
    a_node = _make_node(NodeType.FINAL_OUTPUT)
    store.add_node(d_node)
    store.add_node(a_node)
    k_d = store.register_data_source(d_node.id)
    k_a = store.register_agent_output(a_node.id)
    # Counter is shared, so suffixes are sequential
    num_d = int(k_d.split("_")[1])
    num_a = int(k_a.split("_")[1])
    assert num_a == num_d + 1


def test_resolve_citation_returns_node_id():
    store = ProvenanceStore()
    node = _make_node(NodeType.DATA_READ)
    store.add_node(node)
    key = store.register_data_source(node.id)
    assert store.resolve_citation(key) == node.id


def test_resolve_citation_unknown_key_returns_none():
    store = ProvenanceStore()
    assert store.resolve_citation("d_99") is None


def test_citation_key_for_node_found():
    store = ProvenanceStore()
    node = _make_node(NodeType.DATA_READ)
    store.add_node(node)
    key = store.register_data_source(node.id)
    assert store.citation_key_for_node(node.id) == key


def test_citation_key_for_node_not_found():
    store = ProvenanceStore()
    node = _make_node(NodeType.DATA_READ)
    store.add_node(node)
    assert store.citation_key_for_node(node.id) is None


# ---------------------------------------------------------------------------
# citation_summary
# ---------------------------------------------------------------------------


def test_citation_summary_contains_registered_keys():
    store = ProvenanceStore()
    node = _make_node(NodeType.DATA_READ, "read_report")
    store.add_node(node)
    key = store.register_data_source(node.id)
    summary = store.citation_summary()
    assert key in summary
    assert summary[key]["label"] == "read_report"
    assert summary[key]["type"] == "data_read"


def test_citation_summary_skips_missing_nodes():
    store = ProvenanceStore()
    store._citation_registry["d_1"] = "nonexistent-id"
    summary = store.citation_summary()
    assert "d_1" not in summary


# ---------------------------------------------------------------------------
# Convenience proxies
# ---------------------------------------------------------------------------


def test_store_add_node_proxies_to_graph():
    store = ProvenanceStore()
    node = _make_node(NodeType.INPUT)
    store.add_node(node)
    assert node.id in store.graph.nodes


def test_store_add_edge_proxies_to_graph():
    store = ProvenanceStore()
    src = _make_node(NodeType.INPUT, "src")
    tgt = _make_node(NodeType.TOOL_CALL, "tgt")
    store.add_node(src)
    store.add_node(tgt)
    store.add_edge(src.id, tgt.id, "calls")
    assert len(store.graph.edges) == 1


def test_store_nodes_property():
    store = ProvenanceStore()
    node = _make_node(NodeType.INPUT)
    store.add_node(node)
    assert node.id in store.nodes


def test_store_edges_property():
    store = ProvenanceStore()
    src = _make_node(NodeType.INPUT, "s")
    tgt = _make_node(NodeType.MODEL_REQUEST, "t")
    store.add_node(src)
    store.add_node(tgt)
    store.add_edge(src.id, tgt.id)
    assert len(store.edges) == 1
