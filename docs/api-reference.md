# API Reference

All public symbols are importable directly from `pydantic_ai_provenance`.

---

## Core

### `ProvenanceCapability`

```python
@dataclass
class ProvenanceCapability(AbstractCapability):
    source_tools: list[str] = []
    agent_name: str = "agent"
    inject_citation_instructions: bool = True
```

pydantic-ai `AbstractCapability` that hooks into agent lifecycle events to build the provenance DAG.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `source_tools` | `list[str]` | `[]` | Tool names whose results are raw data sources. Each call gets a unique `d_*` citation key. |
| `agent_name` | `str` | `"agent"` | Label used in graph nodes. |
| `inject_citation_instructions` | `bool` | `True` | Auto-inject citation format instructions into the system prompt. |

**Properties**

| Name | Type | Description |
|---|---|---|
| `store` | `ProvenanceStore` | Available after the run starts. Raises `RuntimeError` if accessed before. |

---

### `ProvenanceStore`

Central registry shared across all agents in a session.

```python
store.register_data_source(node_id: str) -> str
store.register_agent_output(node_id: str) -> str
store.resolve_citation(key: str) -> str | None
store.citation_key_for_node(node_id: str) -> str | None
store.citation_summary() -> dict[str, dict]
```

| Method | Returns | Description |
|---|---|---|
| `register_data_source(node_id)` | `str` | Assign a `d_*` key to a `DATA_READ` node. |
| `register_agent_output(node_id)` | `str` | Assign an `a_*` key to a `FINAL_OUTPUT` node. |
| `resolve_citation(key)` | `str \| None` | Look up the node ID for a citation key. |
| `citation_key_for_node(node_id)` | `str \| None` | Reverse lookup: node ID → citation key. |
| `citation_summary()` | `dict` | Human-readable map of every registered key → node metadata. |

---

## Graph primitives

### `NodeType`

```python
class NodeType(str, Enum):
    INPUT          = "input"
    DATA_READ      = "data_read"
    TOOL_CALL      = "tool_call"
    TOOL_RESULT    = "tool_result"
    MODEL_REQUEST  = "model_request"
    MODEL_RESPONSE = "model_response"
    AGENT_RUN      = "agent_run"
    FINAL_OUTPUT   = "final_output"
```

### `ProvenanceNode`

```python
@dataclass
class ProvenanceNode:
    id: str
    type: NodeType
    label: str
    agent_name: str
    run_id: str
    timestamp: datetime
    data: dict[str, Any]

    @classmethod
    def create(cls, type, label, agent_name, run_id, **data) -> ProvenanceNode: ...
```

### `ProvenanceEdge`

```python
@dataclass
class ProvenanceEdge:
    source_id: str
    target_id: str
    label: str = ""
```

### `ProvenanceGraph`

```python
graph.add_node(node)
graph.add_edge(source_id, target_id, label="")
graph.predecessors(node_id) -> list[ProvenanceNode]
graph.successors(node_id) -> list[ProvenanceNode]
graph.ancestors(node_id) -> set[str]
graph.final_output_nodes() -> list[ProvenanceNode]
graph.source_nodes() -> list[ProvenanceNode]
graph.all_paths_to_sources(node_id) -> list[list[ProvenanceNode]]
```

---

## Attribution

### `attribute_output`

```python
def attribute_output(
    store: ProvenanceStore,
    output_node_id: str | None = None,
) -> AttributionResult
```

Full path-level attribution for one `FINAL_OUTPUT` node. Uses the first `FINAL_OUTPUT` node if `output_node_id` is `None`.

### `attribute_all_outputs`

```python
def attribute_all_outputs(store: ProvenanceStore) -> list[AttributionResult]
```

Attribution for every `FINAL_OUTPUT` node in the graph.

### `AttributionResult`

```python
@dataclass
class AttributionResult:
    output_node: ProvenanceNode
    sources: list[ProvenanceNode]
    paths: list[AttributionPath]

    @property
    def source_labels(self) -> list[str]: ...
    def summary(self) -> str: ...
```

### `AttributionPath`

```python
@dataclass
class AttributionPath:
    source: ProvenanceNode
    path: list[ProvenanceNode]

    @property
    def source_label(self) -> str: ...
    @property
    def hop_count(self) -> int: ...
```

---

## Citations

### `parse_citations`

```python
def parse_citations(text: str) -> list[CitationRef]
```

Extract all `[REF|key1|key2|...]` tags.

### `citation_tag_spans`

```python
def citation_tag_spans(text: str) -> list[tuple[int, int, CitationRef]]
```

Same as `parse_citations` but includes `(start, end)` character positions.

### `strip_inline_citation_tags`

```python
def strip_inline_citation_tags(text: str) -> str
```

Remove all `[REF|…]` tags.

### `strip_inline_citation_tags_preserve_leading_ref_header`

```python
def strip_inline_citation_tags_preserve_leading_ref_header(text: str) -> str
```

Remove inline tags but keep an opening `[REF|…]` block header on the first line.

### `CitationRef`

```python
@dataclass
class CitationRef:
    refs: list[str]   # e.g. ["d_1", "a_2"]
    raw: str          # e.g. "[REF|d_1|a_2]"
```

---

## Verification

### `verify_citations_sync`

```python
def verify_citations_sync(
    text: str,
    store: ProvenanceStore,
    *,
    context_max_chars: int = 720,
) -> CitationVerificationReport
```

Steps 1 (key sanitisation) + 2 (TF-IDF overlap). Returns a `CitationVerificationReport`.

### `strip_unresolvable_citation_keys`

```python
def strip_unresolvable_citation_keys(
    text: str,
    store: ProvenanceStore,
) -> tuple[str, list[CitationKeyFilterResult]]
```

Step 1 only. Returns `(sanitized_text, filter_records)`.

### `claim_source_tfidf_cosine`

```python
def claim_source_tfidf_cosine(
    claim_text: str,
    source_text: str,
    *,
    max_source_chars: int = 96_000,
    chunk_chars: int = 1_200,
    chunk_stride: int = 600,
    max_chunks: int = 400,
) -> float
```

Maximum TF-IDF cosine similarity between a claim and sliding windows of a source. Returns a value in `[0, 1]`.

### `context_before_span`

```python
def context_before_span(
    text: str,
    start: int,
    *,
    max_chars: int = 720,
    max_sentences: int = 1,
) -> str
```

Extract claim context from text immediately before position `start`.

### `entailment_agent`

```python
def entailment_agent(model: Any, *, instructions: str | None = None) -> Agent
```

Build a pydantic-ai agent for Step 3 LLM-based entailment scoring.

### `refine_claim_source_similarities`

```python
def refine_claim_source_similarities(
    records: list[ClaimSourceSimilarity],
    *,
    max_top_n_keys_per_tag: int = 2,
    min_score_for_shared_source: float = 0.3,
) -> list[ClaimSourceSimilarity]
```

Filter similarity records: keep top-N keys per tag and drop weak sources.

### `CitationVerificationReport`

```python
@dataclass
class CitationVerificationReport:
    original_text: str
    text_with_verified_citations: str
    claim_source_similarities: list[ClaimSourceSimilarity]
```

---

## Visualization

| Function | Description |
|---|---|
| `to_mermaid(store)` | Mermaid flowchart string |
| `to_dot(store, graph_name="provenance")` | GraphViz DOT string |
| `to_json(store)` | `dict` with `nodes` and `edges` |
| `to_json_str(store, indent=2)` | JSON string |
