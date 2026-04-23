"""Core graph data structures for the provenance DAG.

Each agent run is recorded as a directed acyclic graph of :class:`ProvenanceNode`
objects connected by labelled :class:`ProvenanceEdge` objects. The graph is stored
in a :class:`ProvenanceGraph` and accessed through :class:`~.store.ProvenanceStore`.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    """The execution stage represented by a :class:`ProvenanceNode`.

    Ordered roughly by when each type appears in a single agent step:

    - ``INPUT`` / ``AGENT_RUN`` — run initialisation
    - ``MODEL_REQUEST`` / ``MODEL_RESPONSE`` — LLM interaction
    - ``TOOL_CALL`` / ``DATA_READ`` / ``TOOL_RESULT`` — tool execution
    - ``FINAL_OUTPUT`` — the text returned to the caller
    """

    INPUT = "input"
    DATA_READ = "data_read"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    AGENT_RUN = "agent_run"
    FINAL_OUTPUT = "final_output"


@dataclass
class ProvenanceNode:
    """A single step in the agent's execution captured as a graph node.

    Prefer :meth:`create` over direct construction — it generates a UUID,
    timestamps the node, and strips ``None`` values from *data*.
    """

    id: str
    type: NodeType
    label: str
    agent_name: str
    run_id: str
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        type: NodeType,
        label: str,
        agent_name: str,
        run_id: str,
        **data: Any,
    ) -> ProvenanceNode:
        """Create a node with a fresh UUID and the current UTC timestamp.

        Keyword arguments in *data* with ``None`` values are excluded from
        the stored dict so callers can pass optional fields without branching.
        """
        return cls(
            id=str(uuid.uuid4()),
            type=type,
            label=label,
            agent_name=agent_name,
            run_id=run_id,
            timestamp=datetime.now(UTC),
            data={k: v for k, v in data.items() if v is not None},
        )


@dataclass
class ProvenanceEdge:
    """A directed edge between two :class:`ProvenanceNode` objects."""

    source_id: str
    target_id: str
    label: str = ""


@dataclass
class ProvenanceGraph:
    """Directed acyclic graph of :class:`ProvenanceNode` objects and :class:`ProvenanceEdge` objects.

    Nodes are keyed by their UUID; edges are stored as an ordered list so
    insertion order is preserved when traversing or serialising the graph.
    """

    nodes: dict[str, ProvenanceNode] = field(default_factory=dict)
    edges: list[ProvenanceEdge] = field(default_factory=list)

    def add_node(self, node: ProvenanceNode) -> None:
        """Insert *node* into the graph, keyed by its UUID.

        If a node with the same ID already exists it will be silently overwritten.

        Args:
            node: The :class:`ProvenanceNode` to register.
        """
        self.nodes[node.id] = node

    def add_edge(self, source_id: str, target_id: str, label: str = "") -> None:
        """Add a directed edge if both endpoint node IDs exist; silently ignored otherwise."""
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append(ProvenanceEdge(source_id, target_id, label))

    def predecessors(self, node_id: str) -> list[ProvenanceNode]:
        """Return nodes that have an edge pointing *to* ``node_id``."""
        ids = {e.source_id for e in self.edges if e.target_id == node_id}
        return [self.nodes[nid] for nid in ids if nid in self.nodes]

    def successors(self, node_id: str) -> list[ProvenanceNode]:
        """Return nodes that ``node_id`` has an edge pointing *to*."""
        ids = {e.target_id for e in self.edges if e.source_id == node_id}
        return [self.nodes[nid] for nid in ids if nid in self.nodes]

    def final_output_nodes(self) -> list[ProvenanceNode]:
        """Return all nodes of type ``FINAL_OUTPUT``."""
        return [n for n in self.nodes.values() if n.type == NodeType.FINAL_OUTPUT]

    def source_nodes(self) -> list[ProvenanceNode]:
        """Return all nodes of type ``INPUT`` or ``DATA_READ``."""
        return [n for n in self.nodes.values() if n.type in (NodeType.INPUT, NodeType.DATA_READ)]

    def ancestors(self, node_id: str) -> set[str]:
        """Return all ancestor node IDs via BFS (does not include ``node_id`` itself)."""
        visited: set[str] = set()
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for pred in self.predecessors(current):
                if pred.id not in visited:
                    visited.add(pred.id)
                    queue.append(pred.id)
        return visited

    def all_paths_to_sources(self, node_id: str) -> list[list[ProvenanceNode]]:
        """Return all paths from source nodes (``INPUT`` / ``DATA_READ``) to ``node_id``.

        Each path is a list ordered from source to ``node_id`` (i.e. root first).
        """
        source_types = {NodeType.INPUT, NodeType.DATA_READ}
        completed: list[list[ProvenanceNode]] = []

        def dfs(current_id: str, path: list[ProvenanceNode]) -> None:
            node = self.nodes.get(current_id)
            if node is None:
                return
            path = [node] + path
            preds = self.predecessors(current_id)
            if not preds or node.type in source_types:
                completed.append(path)
                return
            for pred in preds:
                dfs(pred.id, path)

        dfs(node_id, [])
        return completed
