"""Path-level provenance attribution utilities.

Given a :class:`~.store.ProvenanceStore`, these helpers walk the provenance DAG
backwards from a ``FINAL_OUTPUT`` node to the originating ``INPUT`` and
``DATA_READ`` nodes, returning every path and the unique set of contributing
sources.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai_provenance.graph import NodeType, ProvenanceNode
from pydantic_ai_provenance.store import ProvenanceStore

_SOURCE_TYPES = {NodeType.INPUT, NodeType.DATA_READ}


@dataclass
class AttributionPath:
    """A single chain from a source node to the output node.

    Attributes:
        source: The root node of the path — always an ``INPUT`` or ``DATA_READ``
            node when the path was produced by :func:`attribute_output`.
        path: Ordered list of :class:`~.graph.ProvenanceNode` objects from the
            root source to the final output, inclusive on both ends.
    """

    source: ProvenanceNode
    path: list[ProvenanceNode]

    @property
    def source_label(self) -> str:
        """Human-readable label of the root source node."""
        return self.source.label

    @property
    def hop_count(self) -> int:
        """Number of edges in the path (i.e. ``len(path) - 1``)."""
        return len(self.path) - 1

    def __str__(self) -> str:
        """Render the path as an arrow-separated sequence of node labels."""
        return " → ".join(n.label for n in self.path)


@dataclass
class AttributionResult:
    """Full provenance attribution for a final output node.

    Attributes:
        output_node: The ``FINAL_OUTPUT`` node that was attributed.
        sources: Deduplicated list of ``INPUT`` and ``DATA_READ`` nodes that
            contributed to the output, in insertion order.
        paths: All paths from source nodes to the output, sorted by
            :attr:`~AttributionPath.hop_count` ascending (shortest first).
    """

    output_node: ProvenanceNode
    sources: list[ProvenanceNode]
    paths: list[AttributionPath]

    @property
    def source_labels(self) -> list[str]:
        """Return the human-readable labels of all contributing source nodes."""
        return [s.label for s in self.sources]

    def summary(self) -> str:
        """Render a multi-line human-readable summary of the attribution result.

        The summary lists the output label, each contributing source tagged with
        ``[data]`` or ``[input]``, and every attribution path in hop-count order.

        Returns:
            A newline-separated string suitable for printing to a terminal.
        """
        lines = [f"Output: {self.output_node.label}", ""]
        if not self.sources:
            lines.append("No source nodes found in graph.")
            return "\n".join(lines)
        lines.append(f"Contributing sources ({len(self.sources)}):")
        for src in self.sources:
            tag = "[data]" if src.type == NodeType.DATA_READ else "[input]"
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

    Walks the provenance DAG backwards from the chosen ``FINAL_OUTPUT`` node to
    collect every path that reaches an ``INPUT`` or ``DATA_READ`` node.  Sources
    are deduplicated by node ID while preserving the order in which paths were
    discovered.  Paths are sorted shortest-first.

    Args:
        store: The :class:`~.store.ProvenanceStore` holding the graph.
        output_node_id: UUID of the ``FINAL_OUTPUT`` node to attribute.  When
            ``None``, the first ``FINAL_OUTPUT`` node returned by
            :meth:`~.graph.ProvenanceGraph.final_output_nodes` is used.

    Returns:
        An :class:`AttributionResult` populated with the output node, all
        contributing sources, and all attribution paths.

    Raises:
        ValueError: If *output_node_id* is ``None`` and the graph contains no
            ``FINAL_OUTPUT`` nodes.
        KeyError: If *output_node_id* is provided but does not exist in the graph.
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
    # Dict used as an ordered set: preserves insertion order while deduplicating by node ID.
    unique_sources: dict[str, ProvenanceNode] = {}

    for path in raw_paths:
        if not path:
            continue
        source = path[0]
        if source.type in _SOURCE_TYPES:
            unique_sources[source.id] = source
        attr_paths.append(AttributionPath(source=source, path=path))

    attr_paths.sort(key=lambda p: p.hop_count)

    return AttributionResult(
        output_node=output_node,
        sources=list(unique_sources.values()),
        paths=attr_paths,
    )


def attribute_all_outputs(store: ProvenanceStore) -> list[AttributionResult]:
    """Return attribution for every FINAL_OUTPUT node in the graph.

    Calls :func:`attribute_output` once per ``FINAL_OUTPUT`` node found in
    *store*.  The order of results mirrors the order returned by
    :meth:`~.graph.ProvenanceGraph.final_output_nodes`.

    Args:
        store: The :class:`~.store.ProvenanceStore` holding the graph.

    Returns:
        A list of :class:`AttributionResult` objects, one per ``FINAL_OUTPUT``
        node.  Returns an empty list when the graph has no output nodes.
    """
    graph = store.graph
    return [attribute_output(store, n.id) for n in graph.final_output_nodes()]
