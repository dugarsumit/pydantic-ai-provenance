"""Tests for pydantic_ai_provenance.store."""

from __future__ import annotations

import json

from pydantic_ai_provenance.graph import NodeType, ProvenanceNode
from pydantic_ai_provenance.store import ProvenanceStore


def _make_node(node_type: NodeType, label: str = "node") -> ProvenanceNode:
    return ProvenanceNode.create(type=node_type, label=label, agent_name="agent", run_id="r")


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


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


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
    assert _two_node_store().to_mermaid().startswith("flowchart LR")


def test_to_mermaid_contains_node_labels():
    result = _two_node_store().to_mermaid()
    assert "read_file" in result
    assert "Final answer" in result


def test_to_mermaid_contains_arrow():
    assert "-->" in _two_node_store().to_mermaid()


def test_to_mermaid_contains_classDef():
    assert "classDef" in _two_node_store().to_mermaid()


def test_to_mermaid_empty_store():
    assert ProvenanceStore().to_mermaid().startswith("flowchart LR")


# ---------------------------------------------------------------------------
# to_dot
# ---------------------------------------------------------------------------


def test_to_dot_starts_with_digraph():
    assert _two_node_store().to_dot().startswith("digraph provenance")


def test_to_dot_contains_node_labels():
    result = _two_node_store().to_dot()
    assert "read_file" in result
    assert "Final answer" in result


def test_to_dot_contains_arrow():
    assert "->" in _two_node_store().to_dot()


def test_to_dot_custom_graph_name():
    assert "digraph my_graph" in _two_node_store().to_dot(graph_name="my_graph")


def test_to_dot_ends_with_closing_brace():
    assert _two_node_store().to_dot().strip().endswith("}")


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


def test_to_json_returns_dict():
    assert isinstance(_two_node_store().to_json(), dict)


def test_to_json_has_nodes_and_edges_keys():
    result = _two_node_store().to_json()
    assert "nodes" in result
    assert "edges" in result


def test_to_json_nodes_have_required_fields():
    for node in _two_node_store().to_json()["nodes"]:
        for field in ("id", "type", "label", "agent_name", "run_id", "timestamp"):
            assert field in node


def test_to_json_edges_have_required_fields():
    for edge in _two_node_store().to_json()["edges"]:
        for field in ("source", "target", "label"):
            assert field in edge


def test_to_json_node_count():
    assert len(_two_node_store().to_json()["nodes"]) == 2


def test_to_json_edge_count():
    assert len(_two_node_store().to_json()["edges"]) == 1


# ---------------------------------------------------------------------------
# to_json_str
# ---------------------------------------------------------------------------


def test_to_json_str_returns_valid_json():
    parsed = json.loads(_two_node_store().to_json_str())
    assert "nodes" in parsed
    assert "edges" in parsed


def test_to_json_str_respects_indent():
    result_2 = _two_node_store().to_json_str(indent=2)
    result_4 = _two_node_store().to_json_str(indent=4)
    assert "    " in result_4
    assert result_2 != result_4


# ---------------------------------------------------------------------------
# to_html
# ---------------------------------------------------------------------------


def test_to_html_returns_string():
    assert isinstance(_two_node_store().to_html(), str)


def test_to_html_is_valid_html_document():
    result = _two_node_store().to_html()
    assert "<!DOCTYPE html>" in result
    assert "<html" in result
    assert "</html>" in result


def test_to_html_default_title():
    assert "Provenance Graph" in _two_node_store().to_html()


def test_to_html_custom_title():
    result = _two_node_store().to_html(title="My Custom Title")
    assert "My Custom Title" in result
    assert "Provenance Graph" not in result


def test_to_html_contains_node_labels():
    result = _two_node_store().to_html()
    assert "read_file" in result
    assert "Final answer" in result


def test_to_html_contains_cytoscape_script():
    assert "cytoscape" in _two_node_store().to_html().lower()


def test_to_html_contains_graph_data():
    result = _two_node_store().to_html()
    assert '"nodes"' in result
    assert '"edges"' in result


def test_to_html_escapes_script_tag_sequences():
    store = ProvenanceStore()
    node = ProvenanceNode.create(
        type=NodeType.DATA_READ,
        label="</script><script>alert(1)</script>",
        agent_name="agent",
        run_id="r",
    )
    store.add_node(node)
    result = store.to_html()
    json_start = result.index("const rawGraph = ") + len("const rawGraph = ")
    json_end = result.index(";\n", json_start)
    assert "</script>" not in result[json_start:json_end]


def test_to_html_empty_store():
    result = ProvenanceStore().to_html()
    assert "<!DOCTYPE html>" in result
    assert '"nodes": []' in result or '"nodes":[]' in result
