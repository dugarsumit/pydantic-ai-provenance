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


# ---------------------------------------------------------------------------
# Interactive HTML template (Cytoscape.js + dagre, loaded from CDN)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>__TITLE__</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#ffffff;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;display:flex;height:100vh;overflow:hidden}
    #cy{flex:1;background:#ffffff}
    #sidebar{width:290px;background:#f6f8fa;border-left:1px solid #d0d7de;display:flex;flex-direction:column;min-width:0}
    #title-bar{padding:14px 16px;border-bottom:1px solid #d0d7de}
    #title-bar h1{font-size:15px;font-weight:600;color:#1f2328;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    #graph-stats{font-size:11px;color:#636c76;margin-top:3px}
    #legend{padding:10px 16px;border-bottom:1px solid #d0d7de}
    #legend-title{font-size:11px;font-weight:600;color:#636c76;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
    .legend-item{display:flex;align-items:center;gap:8px;padding:2px 0;font-size:12px;color:#1f2328}
    .legend-swatch{width:12px;height:12px;border-radius:2px;flex-shrink:0}
    #no-selection{padding:20px 16px;color:#636c76;font-size:13px;text-align:center}
    #details{padding:14px 16px;flex:1;overflow-y:auto;display:none}
    #details-label{font-size:14px;font-weight:600;color:#1f2328;margin-bottom:12px;word-break:break-all}
    .dr{margin-bottom:10px}
    .dl{font-size:10px;font-weight:600;color:#636c76;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
    .dv{font-size:12px;color:#1f2328;background:#eaeef2;padding:5px 7px;border-radius:4px;word-break:break-all;white-space:pre-wrap}
    .badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:11px;font-weight:600;color:#fff}
  </style>
</head>
<body>
  <div id="cy"></div>
  <div id="sidebar">
    <div id="title-bar">
      <h1>__TITLE__</h1>
      <div id="graph-stats"></div>
    </div>
    <div id="legend">
      <div id="legend-title">Node types</div>
      <div id="legend-items"></div>
    </div>
    <div id="no-selection">Click a node to inspect it</div>
    <div id="details">
      <div id="details-label"></div>
      <div id="details-body"></div>
    </div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.29.2/cytoscape.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
  <script>
    const rawGraph = __GRAPH_DATA__;

    const NODE_COLORS = {
      input:          "#4A90D9",
      data_read:      "#27AE60",
      tool_call:      "#E67E22",
      tool_result:    "#F39C12",
      model_request:  "#8E44AD",
      model_response: "#9B59B6",
      agent_run:      "#2C3E50",
      final_output:   "#E74C3C",
    };
    const NODE_SHAPES = {
      input:          "round-rectangle",
      data_read:      "barrel",
      tool_call:      "rectangle",
      tool_result:    "rectangle",
      model_request:  "diamond",
      model_response: "diamond",
      agent_run:      "cut-rectangle",
      final_output:   "ellipse",
    };
    const NODE_DISPLAY = {
      input:          "Input",
      data_read:      "Data Read",
      tool_call:      "Tool Call",
      tool_result:    "Tool Result",
      model_request:  "Model Request",
      model_response: "Model Response",
      agent_run:      "Agent Run",
      final_output:   "Final Output",
    };

    // Build legend — node types (only those present in this graph)
    const presentTypes = [...new Set(rawGraph.nodes.map(n => n.type))];
    const legendItems = document.getElementById("legend-items");
    for (const t of Object.keys(NODE_COLORS)) {
      if (!presentTypes.includes(t)) continue;
      const d = document.createElement("div");
      d.className = "legend-item";
      d.innerHTML = `<div class="legend-swatch" style="background:${NODE_COLORS[t]}"></div><span>${NODE_DISPLAY[t] || t}</span>`;
      legendItems.appendChild(d);
    }

    // Build legend — citation key borders (only if any keys exist in this graph)
    const hasDataKey  = rawGraph.nodes.some(n => n.citation_key && n.citation_key.startsWith("d_"));
    const hasAgentKey = rawGraph.nodes.some(n => n.citation_key && n.citation_key.startsWith("a_"));
    if (hasDataKey || hasAgentKey) {
      const ckSection = document.createElement("div");
      ckSection.style.cssText = "margin-top:10px;padding-top:8px;border-top:1px solid #d0d7de";
      ckSection.innerHTML = `<div style="font-size:11px;font-weight:600;color:#636c76;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Citation keys</div>`;
      if (hasDataKey) {
        ckSection.innerHTML += `<div class="legend-item"><div class="legend-swatch" style="background:transparent;border:3px solid #D97706"></div><span>Data source (d_*)</span></div>`;
      }
      if (hasAgentKey) {
        ckSection.innerHTML += `<div class="legend-item"><div class="legend-swatch" style="background:transparent;border:3px solid #2563EB"></div><span>Agent output (a_*)</span></div>`;
      }
      legendItems.appendChild(ckSection);
    }

    document.getElementById("graph-stats").textContent =
      rawGraph.nodes.length + " nodes · " + rawGraph.edges.length + " edges";

    // Build Cytoscape elements
    const elements = [];
    for (const node of rawGraph.nodes) {
      const ck = node.citation_key || null;
      const ckType = ck ? (ck.startsWith("d_") ? "data" : "agent") : null;
      const displayLabel = ck ? node.label + "\\n[" + ck + "]" : node.label;
      elements.push({ data: {
        id: node.id, label: displayLabel, type: node.type,
        agent_name: node.agent_name, run_id: node.run_id,
        timestamp: node.timestamp, extra: node.data || {},
        citation_key: ck, citation_key_type: ckType,
      }});
    }
    for (const edge of rawGraph.edges) {
      elements.push({ data: {
        id: edge.source + "_" + edge.target,
        source: edge.source, target: edge.target, label: edge.label || "",
      }});
    }

    cytoscape.use(cytoscapeDagre);
    const cy = cytoscape({
      container: document.getElementById("cy"),
      elements,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "11px",
            "font-family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            color: "#fff",
            "text-outline-width": 1.5,
            "text-outline-color": "rgba(0,0,0,0.6)",
            "text-wrap": "wrap",
            "text-max-width": "120px",
            width: "label",
            height: "label",
            padding: "12px",
            "background-color": function(ele) { return NODE_COLORS[ele.data("type")] || "#999"; },
            shape: function(ele) { return NODE_SHAPES[ele.data("type")] || "rectangle"; },
            "border-width": 2,
            "border-color": "rgba(255,255,255,0.15)",
          }
        },
        {
          selector: "node[citation_key_type = 'data']",
          style: { "border-width": 3, "border-color": "#D97706", "border-opacity": 1 }
        },
        {
          selector: "node[citation_key_type = 'agent']",
          style: { "border-width": 3, "border-color": "#2563EB", "border-opacity": 1 }
        },
        {
          selector: "node:selected",
          style: { "border-width": 4, "border-color": "#0969da", "border-opacity": 1 }
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": "#8c959f",
            "target-arrow-color": "#8c959f",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": "10px",
            color: "#636c76",
            "text-background-color": "#ffffff",
            "text-background-opacity": 0.85,
            "text-background-padding": "2px",
          }
        },
        {
          selector: "edge:selected",
          style: { "line-color": "#0969da", "target-arrow-color": "#0969da" }
        },
      ],
      layout: { name: "dagre", rankDir: "LR", nodeSep: 45, rankSep: 90, padding: 30 },
    });

    // Details panel
    function esc(s) {
      return String(s)
        .replace(/&/g,"&amp;").replace(/</g,"&lt;")
        .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
    }
    function row(label, value) {
      const v = typeof value === "string" ? value : JSON.stringify(value, null, 2);
      const display = v.length > 500 ? v.slice(0, 500) + "…" : v;
      return `<div class="dr"><div class="dl">${esc(label)}</div><div class="dv">${esc(display)}</div></div>`;
    }

    const noSel   = document.getElementById("no-selection");
    const details = document.getElementById("details");
    const dlabel  = document.getElementById("details-label");
    const dbody   = document.getElementById("details-body");

    cy.on("tap", "node", function(evt) {
      const d = evt.target.data();
      const color = NODE_COLORS[d.type] || "#999";
      noSel.style.display = "none";
      details.style.display = "block";
      const baseLabel = d.citation_key
        ? d.label.replace("\\n[" + d.citation_key + "]", "")
        : d.label;
      dlabel.textContent = baseLabel;
      let html = `<div class="dr"><div class="dl">Type</div><div class="dv"><span class="badge" style="background:${color}">${NODE_DISPLAY[d.type] || d.type}</span></div></div>`;
      if (d.citation_key) {
        html += `<div class="dr"><div class="dl">Citation key</div><div class="dv" style="font-weight:600;color:#0969da">${esc(d.citation_key)}</div></div>`;
      }
      html += row("Agent", d.agent_name);
      html += row("Run ID", d.run_id);
      html += row("Timestamp", d.timestamp);
      html += row("Node ID", d.id);
      for (const [k, v] of Object.entries(d.extra || {})) { html += row(k, v); }
      dbody.innerHTML = html;
    });

    cy.on("tap", function(evt) {
      if (evt.target === cy) {
        noSel.style.display = "block";
        details.style.display = "none";
        cy.elements().unselect();
      }
    });
  </script>
</body>
</html>
"""


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
            d: dict[str, Any] = {
                "id": node.id,
                "type": node.type.value,
                "label": node.label,
                "agent_name": node.agent_name,
                "run_id": node.run_id,
                "timestamp": node.timestamp.isoformat(),
                "data": node.data,
            }
            key = self.citation_key_for_node(node.id)
            if key is not None:
                d["citation_key"] = key
            return d

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

    def to_html(self, title: str = "Provenance Graph") -> str:
        """Render the provenance graph as a self-contained interactive HTML page.

        The page uses Cytoscape.js with dagre layout (loaded from CDN) and requires
        a network connection to render. Click any node to inspect its metadata in the
        sidebar panel.

        Args:
            title: Text shown in the browser tab and sidebar header.

        Returns:
            A complete HTML document as a string. Save to a ``.html`` file or call
            :meth:`open_in_browser` to open it directly.
        """
        # Embed JSON safely inside a <script> block — escape </ to avoid premature
        # script termination if a node label or data value contains that sequence.
        graph_data = json.dumps(self.to_json(), default=str).replace("</", "<\\/")
        return _HTML_TEMPLATE.replace("__TITLE__", title).replace("__GRAPH_DATA__", graph_data)

    def open_in_browser(self, title: str = "Provenance Graph") -> None:
        """Write the interactive HTML visualisation to a temp file and open it.

        The temp file persists until the OS cleans up the temp directory (the browser
        needs it to remain readable). A network connection is required to load the
        Cytoscape.js and dagre CDN scripts.

        Args:
            title: Page title passed to :meth:`to_html`.
        """
        import tempfile
        import webbrowser

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
            f.write(self.to_html(title=title))
            path = f.name
        webbrowser.open(f"file://{path}")
