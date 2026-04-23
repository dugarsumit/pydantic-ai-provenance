"""Graph visualisation: Mermaid flowchart, GraphViz DOT, and JSON export."""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai_provenance.graph import NodeType, ProvenanceNode
from pydantic_ai_provenance.store import ProvenanceStore

# ---------------------------------------------------------------------------
# Mermaid styling
# ---------------------------------------------------------------------------

_MERMAID_SHAPES: dict[NodeType, tuple[str, str]] = {
    NodeType.INPUT: ("([", "])"),
    NodeType.DATA_READ: ("[(", ")]"),
    NodeType.TOOL_CALL: ("[", "]"),
    NodeType.TOOL_RESULT: ("[", "]"),
    NodeType.MODEL_REQUEST: ("{", "}"),
    NodeType.MODEL_RESPONSE: ("{", "}"),
    NodeType.AGENT_RUN: ("[[", "]]"),
    NodeType.FINAL_OUTPUT: ("([", "])"),
}

_MERMAID_STYLES: dict[NodeType, str] = {
    NodeType.INPUT: "fill:#4A90D9,color:#fff,stroke:#2c6fad",
    NodeType.DATA_READ: "fill:#27AE60,color:#fff,stroke:#1a7a43",
    NodeType.TOOL_CALL: "fill:#E67E22,color:#fff,stroke:#b05d10",
    NodeType.TOOL_RESULT: "fill:#F39C12,color:#fff,stroke:#b07a0e",
    NodeType.MODEL_REQUEST: "fill:#8E44AD,color:#fff,stroke:#6a3080",
    NodeType.MODEL_RESPONSE: "fill:#9B59B6,color:#fff,stroke:#7a3d9a",
    NodeType.AGENT_RUN: "fill:#2C3E50,color:#fff,stroke:#1a252f",
    NodeType.FINAL_OUTPUT: "fill:#E74C3C,color:#fff,stroke:#b03020",
}

# ---------------------------------------------------------------------------
# DOT styling
# ---------------------------------------------------------------------------

_DOT_COLORS: dict[NodeType, str] = {
    NodeType.INPUT: "#4A90D9",
    NodeType.DATA_READ: "#27AE60",
    NodeType.TOOL_CALL: "#E67E22",
    NodeType.TOOL_RESULT: "#F39C12",
    NodeType.MODEL_REQUEST: "#8E44AD",
    NodeType.MODEL_RESPONSE: "#9B59B6",
    NodeType.AGENT_RUN: "#2C3E50",
    NodeType.FINAL_OUTPUT: "#E74C3C",
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _short_id(node_id: str) -> str:
    """Return a 16-character hex ID safe for use as a Mermaid/DOT node identifier."""
    return node_id.replace("-", "")[:16]


def _escape_label(text: str) -> str:
    """Escape double-quotes and newlines so the label is safe inside a quoted string."""
    return text.replace('"', "'").replace("\n", " ")


# ---------------------------------------------------------------------------
# Public export functions
# ---------------------------------------------------------------------------


def to_mermaid(store: ProvenanceStore) -> str:
    """Render the provenance graph as a Mermaid flowchart (left-to-right direction)."""
    graph = store.graph
    lines = ["flowchart LR"]

    style_classes: dict[NodeType, list[str]] = {t: [] for t in NodeType}

    for node in graph.nodes.values():
        sid = _short_id(node.id)
        open_b, close_b = _MERMAID_SHAPES.get(node.type, ("[", "]"))
        label = _escape_label(node.label)
        lines.append(f'    {sid}{open_b}"{label}"{close_b}')
        style_classes[node.type].append(sid)

    lines.append("")
    for edge in graph.edges:
        src = _short_id(edge.source_id)
        tgt = _short_id(edge.target_id)
        lbl = f"|{edge.label}|" if edge.label else ""
        lines.append(f"    {src} --{lbl}--> {tgt}")

    lines.append("")
    for node_type, ids in style_classes.items():
        if not ids:
            continue
        class_name = node_type.value
        style = _MERMAID_STYLES[node_type]
        lines.append(f"    classDef {class_name} {style}")
        lines.append(f"    class {','.join(ids)} {class_name}")

    return "\n".join(lines)


def to_dot(store: ProvenanceStore, graph_name: str = "provenance") -> str:
    """Render the provenance graph as a Graphviz DOT string."""
    graph = store.graph
    lines = [
        f"digraph {graph_name} {{",
        "    rankdir=LR;",
        "    node [style=filled, fontcolor=white, fontname=Helvetica];",
    ]

    for node in graph.nodes.values():
        sid = _short_id(node.id)
        label = _escape_label(node.label)
        color = _DOT_COLORS.get(node.type, "#999999")
        shape = "cylinder" if node.type == NodeType.DATA_READ else "box"
        lines.append(f'    {sid} [label="{label}", fillcolor="{color}", shape={shape}];')

    lines.append("")
    for edge in graph.edges:
        src = _short_id(edge.source_id)
        tgt = _short_id(edge.target_id)
        lbl = _escape_label(edge.label)
        lines.append(f'    {src} -> {tgt} [label="{lbl}"];')

    lines.append("}")
    return "\n".join(lines)


def to_json(store: ProvenanceStore) -> dict[str, Any]:
    """Serialise the provenance graph to a JSON-compatible dict with ``nodes`` and ``edges`` keys."""
    graph = store.graph

    def _node_dict(node: ProvenanceNode) -> dict[str, Any]:
        """Serialise a single :class:`~.graph.ProvenanceNode` to a JSON-compatible dict.

        The ``timestamp`` field is converted to an ISO-8601 string.  All other
        fields are native Python types already safe for ``json.dumps``.

        Args:
            node: The node to serialise.

        Returns:
            A dictionary with keys: ``id``, ``type``, ``label``, ``agent_name``,
            ``run_id``, ``timestamp``, and ``data``.
        """
        return {
            "id": node.id,
            "type": node.type.value,
            "label": node.label,
            "agent_name": node.agent_name,
            "run_id": node.run_id,
            "timestamp": node.timestamp.isoformat(),
            "data": node.data,
        }

    return {
        "nodes": [_node_dict(n) for n in graph.nodes.values()],
        "edges": [{"source": e.source_id, "target": e.target_id, "label": e.label} for e in graph.edges],
    }


def to_json_str(store: ProvenanceStore, indent: int = 2) -> str:
    """Serialise the provenance graph to a JSON string. See :func:`to_json` for the schema."""
    return json.dumps(to_json(store), indent=indent, default=str)
