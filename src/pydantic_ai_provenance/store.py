"""Central provenance store and context-variable used for subagent linkage."""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai_provenance.citations import _KEY_PREFIX_AGENT, _KEY_PREFIX_DATA
from pydantic_ai_provenance.graph import NodeType, ProvenanceEdge, ProvenanceGraph, ProvenanceNode

# Propagates (current_store, parent_tool_call_node_id) into subagent runs so they
# reuse the parent's store and can be linked back to the tool call that spawned them.
# None when no provenance context is active (top-level run, no capability attached).
_PROVENANCE_CTX: ContextVar[tuple[ProvenanceStore, str | None] | None] = ContextVar(
    "pydantic_ai_provenance", default=None
)

# Node data keys included in citation_summary() output.
_SUMMARY_NODE_DATA_KEYS = frozenset({"file_path", "output", "tool_name"})

# ---------------------------------------------------------------------------
# Visualisation constants
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


def _short_id(node_id: str) -> str:
    return node_id.replace("-", "")[:16]


def _escape_label(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


@dataclass
class ProvenanceStore:
    """Central registry that holds the provenance graph and the citation key → node mapping.

    The citation registry maps short human-readable keys (``d_1``, ``a_2``, …) to
    node IDs in the graph.  All metadata lives in the corresponding
    :class:`~.graph.ProvenanceNode`; the key is only ever embedded in tool/agent
    payloads and in the model's final output as ``[REF|<key>]`` tags.

    A single store is shared across all agents in a session so that:

    - multiple reads of the same resource each get a distinct key (``d_1``, ``d_2``);
    - subagent citations resolve back to nodes created by parent agents.
    """

    graph: ProvenanceGraph = field(default_factory=ProvenanceGraph)
    _citation_registry: dict[str, str] = field(default_factory=dict)
    _citation_counter: int = field(default=0)

    # ------------------------------------------------------------------
    # Graph proxy helpers
    # ------------------------------------------------------------------

    def add_node(self, node: ProvenanceNode) -> None:
        """Proxy for :meth:`~.graph.ProvenanceGraph.add_node` on the underlying graph.

        Args:
            node: The :class:`~.graph.ProvenanceNode` to register.
        """
        self.graph.add_node(node)

    def add_edge(self, source_id: str, target_id: str, label: str = "") -> None:
        """Proxy for :meth:`~.graph.ProvenanceGraph.add_edge` on the underlying graph.

        The edge is silently ignored if either endpoint ID is not present in the
        graph.

        Args:
            source_id: UUID of the source node.
            target_id: UUID of the target node.
            label: Optional human-readable edge label (e.g. ``"feeds_into"``).
        """
        self.graph.add_edge(source_id, target_id, label)

    # ------------------------------------------------------------------
    # Citation registry
    # ------------------------------------------------------------------

    def register_data_source(self, node_id: str) -> str:
        """Assign a unique ``d_*`` citation key to a ``DATA_READ`` node.

        Each call — even for the same underlying resource — produces a distinct key
        so that multiple reads are tracked independently.

        Returns the key (e.g. ``"d_1"``) to embed in the tool result shown to the LLM.
        """
        self._citation_counter += 1
        key = f"{_KEY_PREFIX_DATA}_{self._citation_counter}"
        self._citation_registry[key] = node_id
        return key

    def register_agent_output(self, node_id: str) -> str:
        """Assign a unique ``a_*`` citation key to a ``FINAL_OUTPUT`` node.

        Returns the key (e.g. ``"a_1"``) to embed in the tool result shown to the
        parent agent's LLM.
        """
        self._citation_counter += 1
        key = f"{_KEY_PREFIX_AGENT}_{self._citation_counter}"
        self._citation_registry[key] = node_id
        return key

    def resolve_citation(self, key: str) -> str | None:
        """Return the node ID for *key*, or ``None`` if the key is not registered."""
        return self._citation_registry.get(key)

    def citation_key_for_node(self, node_id: str) -> str | None:
        """Return the citation key registered for *node_id*, or ``None`` if absent."""
        for key, nid in self._citation_registry.items():
            if nid == node_id:
                return key
        return None

    def citation_summary(self) -> dict[str, dict]:
        """Return a human-readable mapping of every citation key to its node metadata.

        Useful for post-run inspection: which file or agent output each key refers to
        without having to walk the full graph. Includes ``file_path``, ``output``, and
        ``tool_name`` from the node's data dict where present.
        """
        summary: dict[str, dict] = {}
        for key, node_id in self._citation_registry.items():
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            summary[key] = {
                "node_id": node_id,
                "type": node.type.value,
                "label": node.label,
                "agent_name": node.agent_name,
                "run_id": node.run_id,
                **{k: v for k, v in node.data.items() if k in _SUMMARY_NODE_DATA_KEYS},
            }
        return summary

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> dict[str, ProvenanceNode]:
        """Direct reference to the graph's node dictionary, keyed by node UUID."""
        return self.graph.nodes

    @property
    def edges(self) -> list[ProvenanceEdge]:
        """Direct reference to the graph's edge list in insertion order."""
        return self.graph.edges

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def to_mermaid(self) -> str:
        """Render the provenance graph as a Mermaid flowchart (left-to-right direction).

        Paste the result into `mermaid.live <https://mermaid.live>`_ to visualise
        the execution DAG.
        """
        lines = ["flowchart LR"]
        style_classes: dict[NodeType, list[str]] = {t: [] for t in NodeType}

        for node in self.graph.nodes.values():
            sid = _short_id(node.id)
            open_b, close_b = _MERMAID_SHAPES.get(node.type, ("[", "]"))
            label = _escape_label(node.label)
            lines.append(f'    {sid}{open_b}"{label}"{close_b}')
            style_classes[node.type].append(sid)

        lines.append("")
        for edge in self.graph.edges:
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

    def to_dot(self, graph_name: str = "provenance") -> str:
        """Render the provenance graph as a Graphviz DOT string.

        Args:
            graph_name: Name embedded in the ``digraph`` declaration.
        """
        lines = [
            f"digraph {graph_name} {{",
            "    rankdir=LR;",
            "    node [style=filled, fontcolor=white, fontname=Helvetica];",
        ]

        for node in self.graph.nodes.values():
            sid = _short_id(node.id)
            label = _escape_label(node.label)
            color = _DOT_COLORS.get(node.type, "#999999")
            shape = "cylinder" if node.type == NodeType.DATA_READ else "box"
            lines.append(f'    {sid} [label="{label}", fillcolor="{color}", shape={shape}];')

        lines.append("")
        for edge in self.graph.edges:
            src = _short_id(edge.source_id)
            tgt = _short_id(edge.target_id)
            lbl = _escape_label(edge.label)
            lines.append(f'    {src} -> {tgt} [label="{lbl}"];')

        lines.append("}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Serialise the provenance graph to a JSON-compatible dict.

        Returns a dict with ``nodes`` and ``edges`` keys.
        """

        def _node_dict(node: ProvenanceNode) -> dict[str, Any]:
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
            "nodes": [_node_dict(n) for n in self.graph.nodes.values()],
            "edges": [{"source": e.source_id, "target": e.target_id, "label": e.label} for e in self.graph.edges],
        }

    def to_json_str(self, indent: int = 2) -> str:
        """Serialise the provenance graph to a JSON string.

        Args:
            indent: Indentation level passed to :func:`json.dumps`.
        """
        return json.dumps(self.to_json(), indent=indent, default=str)
