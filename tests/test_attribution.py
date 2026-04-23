"""Tests for pydantic_ai_provenance.attribution."""

from __future__ import annotations

import pytest

from pydantic_ai_provenance.attribution import (
    AttributionPath,
    AttributionResult,
    attribute_all_outputs,
    attribute_output,
)
from pydantic_ai_provenance.graph import NodeType, ProvenanceNode
from pydantic_ai_provenance.store import ProvenanceStore


def _make_node(node_type: NodeType, label: str) -> ProvenanceNode:
    return ProvenanceNode.create(
        type=node_type, label=label, agent_name="agent", run_id="r"
    )


def _build_linear_store() -> tuple[ProvenanceStore, str, str]:
    """INPUT → MODEL_REQUEST → FINAL_OUTPUT. Returns (store, input_id, output_id)."""
    store = ProvenanceStore()
    inp = _make_node(NodeType.INPUT, "User input")
    req = _make_node(NodeType.MODEL_REQUEST, "Model request")
    out = _make_node(NodeType.FINAL_OUTPUT, "Final output")
    for n in (inp, req, out):
        store.add_node(n)
    store.add_edge(inp.id, req.id, "feeds_into")
    store.add_edge(req.id, out.id, "produces")
    return store, inp.id, out.id


# ---------------------------------------------------------------------------
# attribute_output
# ---------------------------------------------------------------------------


def test_attribute_output_raises_with_no_final_output():
    store = ProvenanceStore()
    with pytest.raises(ValueError, match="No FINAL_OUTPUT"):
        attribute_output(store)


def test_attribute_output_simple_linear_path():
    store, _, _ = _build_linear_store()
    result = attribute_output(store)
    assert result.output_node.type == NodeType.FINAL_OUTPUT
    assert len(result.paths) >= 1


def test_attribute_output_by_explicit_id():
    store, _, out_id = _build_linear_store()
    result = attribute_output(store, output_node_id=out_id)
    assert result.output_node.id == out_id


def test_attribute_output_paths_start_from_source():
    store, _, _ = _build_linear_store()
    result = attribute_output(store)
    first_path = result.paths[0]
    assert first_path.path[0].type in (NodeType.INPUT, NodeType.DATA_READ)


def test_attribute_output_sources_list():
    store, _, _ = _build_linear_store()
    result = attribute_output(store)
    assert len(result.sources) == 1
    assert result.sources[0].type == NodeType.INPUT


# ---------------------------------------------------------------------------
# attribute_all_outputs
# ---------------------------------------------------------------------------


def test_attribute_all_outputs_empty_store():
    store = ProvenanceStore()
    assert attribute_all_outputs(store) == []


def test_attribute_all_outputs_two_outputs():
    store = ProvenanceStore()
    inp = _make_node(NodeType.INPUT, "inp")
    out1 = _make_node(NodeType.FINAL_OUTPUT, "out1")
    out2 = _make_node(NodeType.FINAL_OUTPUT, "out2")
    for n in (inp, out1, out2):
        store.add_node(n)
    store.add_edge(inp.id, out1.id, "produces")
    store.add_edge(inp.id, out2.id, "produces")

    results = attribute_all_outputs(store)
    assert len(results) == 2
    output_labels = {r.output_node.label for r in results}
    assert output_labels == {"out1", "out2"}


# ---------------------------------------------------------------------------
# AttributionPath
# ---------------------------------------------------------------------------


def test_attribution_path_str():
    a = _make_node(NodeType.INPUT, "Start")
    b = _make_node(NodeType.TOOL_RESULT, "Middle")
    c = _make_node(NodeType.FINAL_OUTPUT, "End")
    path = AttributionPath(source=a, path=[a, b, c])
    assert str(path) == "Start → Middle → End"


def test_attribution_path_hop_count():
    a = _make_node(NodeType.INPUT, "a")
    b = _make_node(NodeType.MODEL_REQUEST, "b")
    c = _make_node(NodeType.FINAL_OUTPUT, "c")
    path = AttributionPath(source=a, path=[a, b, c])
    assert path.hop_count == 2


def test_attribution_path_source_label():
    a = _make_node(NodeType.DATA_READ, "report.txt")
    path = AttributionPath(source=a, path=[a])
    assert path.source_label == "report.txt"


# ---------------------------------------------------------------------------
# AttributionResult
# ---------------------------------------------------------------------------


def test_attribution_result_source_labels():
    store, _, _ = _build_linear_store()
    result = attribute_output(store)
    assert isinstance(result.source_labels, list)
    assert len(result.source_labels) > 0


def test_attribution_result_summary_no_sources():
    store = ProvenanceStore()
    out = _make_node(NodeType.FINAL_OUTPUT, "Out")
    store.add_node(out)
    result = AttributionResult(output_node=out, sources=[], paths=[])
    summary = result.summary()
    assert "No source nodes" in summary


def test_attribution_result_summary_with_sources():
    store, _, _ = _build_linear_store()
    result = attribute_output(store)
    summary = result.summary()
    assert "Output:" in summary
    assert "Contributing sources" in summary
    assert "Attribution paths" in summary
