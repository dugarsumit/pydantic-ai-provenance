from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from .citations import KEY_PREFIX_AGENT, KEY_PREFIX_DATA
from .graph import ProvenanceEdge, ProvenanceGraph, ProvenanceNode


# Stores (current_store, parent_tool_call_node_id) for subagent linkage.
# None means no provenance context is active (top-level, no capability yet).
_PROVENANCE_CTX: ContextVar[tuple[ProvenanceStore, str | None] | None] = ContextVar(
    "pydantic_ai_provenance", default=None
)


@dataclass
class ProvenanceStore:
    graph: ProvenanceGraph = field(default_factory=ProvenanceGraph)

    # Unified citation registry: citation_key → node_id.
    #
    # Keys are short identifiers like "d_1" or "a_2" assigned at registration.
    # Tool/agent payloads use a leading ``[REF|<key>]`` line; inline tags in the
    # body are stripped when formatting. The final user-facing reply may contain
    # ``[REF|…]`` tags parsed for ``cited_in`` edges.
    #
    # All metadata (file path, agent name, run_id, etc.) lives in the
    # corresponding ProvenanceNode — resolved via graph.nodes[node_id].
    #
    # Shared across all agents in a session, so:
    #  - multiple reads of the same resource each get a distinct key (d_1, d_2)
    #  - subagent citations resolve to nodes created by parent agents
    _citation_registry: dict[str, str] = field(default_factory=dict)
    _citation_counter: int = field(default=0)

    def add_node(self, node: ProvenanceNode) -> None:
        self.graph.add_node(node)

    def add_edge(self, source_id: str, target_id: str, label: str = "") -> None:
        self.graph.add_edge(source_id, target_id, label)

    # --- Citation registry ---

    def register_data_source(self, node_id: str) -> str:
        """Assign a unique citation key to a DATA_READ node and register it.

        Each call — even for the same underlying resource — produces a distinct key,
        so multiple reads are tracked independently.

        Returns the citation key (e.g. "d_1") to embed in the formatted
        tool result shown to the LLM.
        """
        self._citation_counter += 1
        key = f"{KEY_PREFIX_DATA}_{self._citation_counter}"
        self._citation_registry[key] = node_id
        return key

    def register_agent_output(self, node_id: str) -> str:
        """Assign a unique citation key to a FINAL_OUTPUT node and register it.

        Returns the citation key (e.g. "a_1") to embed in the formatted
        tool result shown to the parent LLM.
        """
        self._citation_counter += 1
        key = f"{KEY_PREFIX_AGENT}_{self._citation_counter}"
        self._citation_registry[key] = node_id
        return key

    def resolve_citation(self, key: str) -> str | None:
        """Return the node_id for a citation key, or None if not registered."""
        return self._citation_registry.get(key)

    def citation_key_for_node(self, node_id: str) -> str | None:
        """Return the citation key registered for this node, if any."""
        for key, nid in self._citation_registry.items():
            if nid == node_id:
                return key
        return None

    def citation_summary(self) -> dict[str, dict]:
        """Return a human-readable mapping of every citation key to its node metadata.

        Useful for post-run inspection: which file / agent output each key refers to,
        without having to walk the full graph.
        """
        result: dict[str, dict] = {}
        for key, node_id in self._citation_registry.items():
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            result[key] = {
                "node_id": node_id,
                "type": node.type.value,
                "label": node.label,
                "agent_name": node.agent_name,
                "run_id": node.run_id,
                **{k: v for k, v in node.data.items() if k in ("file_path", "output", "tool_name")},
            }
        return result

    # --- Convenience accessors ---

    @property
    def nodes(self) -> dict[str, ProvenanceNode]:
        return self.graph.nodes

    @property
    def edges(self) -> list[ProvenanceEdge]:
        return self.graph.edges
