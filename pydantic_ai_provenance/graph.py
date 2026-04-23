from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class NodeType(str, Enum):
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
        return cls(
            id=str(uuid.uuid4()),
            type=type,
            label=label,
            agent_name=agent_name,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            data={k: v for k, v in data.items() if v is not None},
        )


@dataclass
class ProvenanceEdge:
    source_id: str
    target_id: str
    label: str = ""


@dataclass
class ProvenanceGraph:
    nodes: dict[str, ProvenanceNode] = field(default_factory=dict)
    edges: list[ProvenanceEdge] = field(default_factory=list)

    def add_node(self, node: ProvenanceNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, source_id: str, target_id: str, label: str = "") -> None:
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append(ProvenanceEdge(source_id, target_id, label))

    def predecessors(self, node_id: str) -> list[ProvenanceNode]:
        ids = {e.source_id for e in self.edges if e.target_id == node_id}
        return [self.nodes[nid] for nid in ids if nid in self.nodes]

    def successors(self, node_id: str) -> list[ProvenanceNode]:
        ids = {e.target_id for e in self.edges if e.source_id == node_id}
        return [self.nodes[nid] for nid in ids if nid in self.nodes]

    def final_output_nodes(self) -> list[ProvenanceNode]:
        return [n for n in self.nodes.values() if n.type == NodeType.FINAL_OUTPUT]

    def source_nodes(self) -> list[ProvenanceNode]:
        return [n for n in self.nodes.values() if n.type in (NodeType.INPUT, NodeType.DATA_READ)]

    def ancestors(self, node_id: str) -> set[str]:
        """Return all ancestor node IDs via BFS."""
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
        """Return all paths from source nodes (INPUT/DATA_READ) to node_id, reversed."""
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
