"""Central provenance store and context-variable used for subagent linkage."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from pydantic_ai_provenance.citations import KEY_PREFIX_AGENT, KEY_PREFIX_DATA
from pydantic_ai_provenance.graph import ProvenanceEdge, ProvenanceGraph, ProvenanceNode

# Propagates (current_store, parent_tool_call_node_id) into subagent runs so they
# reuse the parent's store and can be linked back to the tool call that spawned them.
# None when no provenance context is active (top-level run, no capability attached).
_PROVENANCE_CTX: ContextVar[tuple[ProvenanceStore, str | None] | None] = ContextVar(
    "pydantic_ai_provenance", default=None
)

# Node data keys included in citation_summary() output.
_SUMMARY_NODE_DATA_KEYS = frozenset({"file_path", "output", "tool_name"})


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
        key = f"{KEY_PREFIX_DATA}_{self._citation_counter}"
        self._citation_registry[key] = node_id
        return key

    def register_agent_output(self, node_id: str) -> str:
        """Assign a unique ``a_*`` citation key to a ``FINAL_OUTPUT`` node.

        Returns the key (e.g. ``"a_1"``) to embed in the tool result shown to the
        parent agent's LLM.
        """
        self._citation_counter += 1
        key = f"{KEY_PREFIX_AGENT}_{self._citation_counter}"
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
