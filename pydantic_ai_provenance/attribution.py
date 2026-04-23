from __future__ import annotations

from dataclasses import dataclass

from .graph import NodeType, ProvenanceGraph, ProvenanceNode
from .store import ProvenanceStore

_SOURCE_TYPES = {NodeType.INPUT, NodeType.FILE_READ}


@dataclass
class AttributionPath:
    """A single chain from a source node to the output node."""

    source: ProvenanceNode
    path: list[ProvenanceNode]

    @property
    def source_label(self) -> str:
        return self.source.label

    @property
    def hop_count(self) -> int:
        return len(self.path) - 1

    def __str__(self) -> str:
        return " → ".join(n.label for n in self.path)


@dataclass
class AttributionResult:
    """Full provenance attribution for a final output node."""

    output_node: ProvenanceNode
    sources: list[ProvenanceNode]
    paths: list[AttributionPath]

    @property
    def source_labels(self) -> list[str]:
        return [s.label for s in self.sources]

    def summary(self) -> str:
        lines = [f"Output: {self.output_node.label}", ""]
        if not self.sources:
            lines.append("No source nodes found in graph.")
            return "\n".join(lines)
        lines.append(f"Contributing sources ({len(self.sources)}):")
        for src in self.sources:
            tag = "[file/data]" if src.type == NodeType.FILE_READ else "[input]"
            lines.append(f"  {tag} {src.label}")
        lines.append("")
        lines.append(f"Attribution paths ({len(self.paths)}):")
        for i, p in enumerate(self.paths, 1):
            lines.append(f"  {i}. {p}")
        return "\n".join(lines)


def attribute_output(
    store: ProvenanceStore,
    output_node_id: str | None = None,
) -> AttributionResult:
    """Return full path-level attribution for a final output node.

    If output_node_id is None, uses the first FINAL_OUTPUT node in the graph.
    """
    graph = store.graph

    if output_node_id is None:
        candidates = graph.final_output_nodes()
        if not candidates:
            raise ValueError("No FINAL_OUTPUT node found in the graph.")
        output_node = candidates[0]
    else:
        output_node = graph.nodes[output_node_id]

    raw_paths = graph.all_paths_to_sources(output_node.id)

    attr_paths: list[AttributionPath] = []
    seen_sources: dict[str, ProvenanceNode] = {}

    for path in raw_paths:
        if not path:
            continue
        source = path[0]
        if source.type in _SOURCE_TYPES:
            seen_sources[source.id] = source
        attr_paths.append(AttributionPath(source=source, path=path))

    attr_paths.sort(key=lambda p: p.hop_count)

    return AttributionResult(
        output_node=output_node,
        sources=list(seen_sources.values()),
        paths=attr_paths,
    )


def attribute_all_outputs(store: ProvenanceStore) -> list[AttributionResult]:
    """Return attribution for every FINAL_OUTPUT node in the graph."""
    graph = store.graph
    return [attribute_output(store, n.id) for n in graph.final_output_nodes()]
