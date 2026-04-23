"""Tests for pydantic_ai_provenance.graph."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pydantic_ai_provenance.graph import (
    NodeType,
    ProvenanceEdge,
    ProvenanceGraph,
    ProvenanceNode,
)


# ---------------------------------------------------------------------------
# NodeType
# ---------------------------------------------------------------------------


def test_node_type_values():
    assert NodeType.INPUT.value == "input"
    assert NodeType.DATA_READ.value == "data_read"
    assert NodeType.TOOL_CALL.value == "tool_call"
    assert NodeType.TOOL_RESULT.value == "tool_result"
    assert NodeType.MODEL_REQUEST.value == "model_request"
    assert NodeType.MODEL_RESPONSE.value == "model_response"
    assert NodeType.AGENT_RUN.value == "agent_run"
    assert NodeType.FINAL_OUTPUT.value == "final_output"


# ---------------------------------------------------------------------------
# ProvenanceNode
# ---------------------------------------------------------------------------


def test_provenance_node_create_sets_uuid():
    node = ProvenanceNode.create(
        type=NodeType.INPUT, label="test", agent_name="a", run_id="r"
    )
    assert len(node.id) > 0
    assert "-" in node.id


def test_provenance_node_create_sets_timestamp():
    before = datetime.now(timezone.utc)
    node = ProvenanceNode.create(
        type=NodeType.INPUT, label="test", agent_name="a", run_id="r"
    )
    after = datetime.now(timezone.utc)
    assert before <= node.timestamp <= after


def test_provenance_node_create_stores_data():
    node = ProvenanceNode.create(
        type=NodeType.DATA_READ,
        label="read",
        agent_name="a",
        run_id="r",
        file_path="/tmp/x.txt",
    )
    assert node.data["file_path"] == "/tmp/x.txt"


def test_provenance_node_create_excludes_none_data():
    node = ProvenanceNode.create(
        type=NodeType.TOOL_CALL,
        label="call",
        agent_name="a",
        run_id="r",
        file_path=None,
        tool_name="my_tool",
    )
    assert "file_path" not in node.data
    assert node.data["tool_name"] == "my_tool"


# ---------------------------------------------------------------------------
# ProvenanceEdge
# ---------------------------------------------------------------------------


def test_provenance_edge_default_label():
    edge = ProvenanceEdge(source_id="a", target_id="b")
    assert edge.label == ""


def test_provenance_edge_with_label():
    edge = ProvenanceEdge(source_id="a", target_id="b", label="feeds_into")
    assert edge.label == "feeds_into"


# ---------------------------------------------------------------------------
# ProvenanceGraph
# ---------------------------------------------------------------------------


def _make_node(label: str, node_type: NodeType = NodeType.INPUT) -> ProvenanceNode:
    return ProvenanceNode.create(
        type=node_type, label=label, agent_name="agent", run_id="r"
    )


def test_graph_add_and_retrieve_node():
    g = ProvenanceGraph()
    node = _make_node("input")
    g.add_node(node)
    assert node.id in g.nodes
    assert g.nodes[node.id] is node


def test_graph_add_edge_valid():
    g = ProvenanceGraph()
    src = _make_node("src")
    tgt = _make_node("tgt")
    g.add_node(src)
    g.add_node(tgt)
    g.add_edge(src.id, tgt.id, "calls")
    assert len(g.edges) == 1
    assert g.edges[0].source_id == src.id
    assert g.edges[0].target_id == tgt.id
    assert g.edges[0].label == "calls"


def test_graph_add_edge_missing_source_is_ignored():
    g = ProvenanceGraph()
    tgt = _make_node("tgt")
    g.add_node(tgt)
    g.add_edge("nonexistent", tgt.id)
    assert len(g.edges) == 0


def test_graph_add_edge_missing_target_is_ignored():
    g = ProvenanceGraph()
    src = _make_node("src")
    g.add_node(src)
    g.add_edge(src.id, "nonexistent")
    assert len(g.edges) == 0


def test_predecessors():
    g = ProvenanceGraph()
    a = _make_node("a")
    b = _make_node("b")
    g.add_node(a)
    g.add_node(b)
    g.add_edge(a.id, b.id)
    assert g.predecessors(b.id) == [a]
    assert g.predecessors(a.id) == []


def test_successors():
    g = ProvenanceGraph()
    a = _make_node("a")
    b = _make_node("b")
    g.add_node(a)
    g.add_node(b)
    g.add_edge(a.id, b.id)
    assert g.successors(a.id) == [b]
    assert g.successors(b.id) == []


def test_final_output_nodes():
    g = ProvenanceGraph()
    inp = _make_node("inp", NodeType.INPUT)
    out = _make_node("out", NodeType.FINAL_OUTPUT)
    g.add_node(inp)
    g.add_node(out)
    final = g.final_output_nodes()
    assert final == [out]


def test_source_nodes():
    g = ProvenanceGraph()
    inp = _make_node("inp", NodeType.INPUT)
    data = _make_node("data", NodeType.DATA_READ)
    tool = _make_node("tool", NodeType.TOOL_CALL)
    for n in (inp, data, tool):
        g.add_node(n)
    sources = g.source_nodes()
    assert inp in sources
    assert data in sources
    assert tool not in sources


def test_ancestors_bfs():
    g = ProvenanceGraph()
    a = _make_node("a")
    b = _make_node("b")
    c = _make_node("c")
    g.add_node(a)
    g.add_node(b)
    g.add_node(c)
    g.add_edge(a.id, b.id)
    g.add_edge(b.id, c.id)
    ancestors = g.ancestors(c.id)
    assert a.id in ancestors
    assert b.id in ancestors
    assert c.id not in ancestors


def test_all_paths_to_sources_linear():
    g = ProvenanceGraph()
    src = _make_node("src", NodeType.DATA_READ)
    mid = _make_node("mid", NodeType.TOOL_RESULT)
    out = _make_node("out", NodeType.FINAL_OUTPUT)
    for n in (src, mid, out):
        g.add_node(n)
    g.add_edge(src.id, mid.id)
    g.add_edge(mid.id, out.id)

    paths = g.all_paths_to_sources(out.id)
    assert len(paths) == 1
    labels = [n.label for n in paths[0]]
    assert labels == ["src", "mid", "out"]


def test_all_paths_to_sources_two_sources():
    g = ProvenanceGraph()
    s1 = _make_node("s1", NodeType.DATA_READ)
    s2 = _make_node("s2", NodeType.DATA_READ)
    out = _make_node("out", NodeType.FINAL_OUTPUT)
    for n in (s1, s2, out):
        g.add_node(n)
    g.add_edge(s1.id, out.id)
    g.add_edge(s2.id, out.id)

    paths = g.all_paths_to_sources(out.id)
    assert len(paths) == 2
    start_labels = {p[0].label for p in paths}
    assert start_labels == {"s1", "s2"}
