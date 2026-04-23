# pydantic-ai-provenance

**Provenance tracking and citation verification for [pydantic-ai](https://ai.pydantic.dev/) agents.**

Attach `ProvenanceCapability` to any pydantic-ai agent and get:

- A **full execution DAG** — every tool call, model request, and response linked in a directed acyclic graph.
- **Automatic citation keys** (`d_1`, `d_2`, `a_1`, …) injected into source tool results so the LLM can cite them inline.
- **Multi-agent attribution** — subagent outputs propagate through a shared store via `contextvars`, enabling transitive citation resolution across agent boundaries.
- **Citation verification** — TF-IDF cosine overlap (Step 2) and optional LLM entailment (Step 3) to validate every `[REF|…]` tag in the final output.
- **Graph visualisation** — export as Mermaid, GraphViz DOT, or JSON.

---

## Installation

```bash
pip install pydantic-ai-provenance
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pydantic-ai-provenance
```

**Requirements:** Python ≥ 3.10, pydantic-ai ≥ 1.80.

---

## Quick start

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai_provenance import (
    ProvenanceCapability,
    attribute_output,
    to_mermaid,
    verify_citations_sync,
)

provenance = ProvenanceCapability(
    agent_name="summariser",
    source_tools=["read_file"],   # tools whose results are raw data sources
)

agent = Agent(
    "anthropic:claude-sonnet-4-6",
    capabilities=[provenance],
    system_prompt="Summarise the content of files.",
)

@agent.tool_plain
def read_file(path: str) -> str:
    return open(path).read()

async def main():
    result = await agent.run("Read report.txt and summarise it.")

    store = provenance.store

    # Path-level attribution
    print(attribute_output(store).summary())

    # Mermaid diagram
    print(to_mermaid(store))

    # Citation verification (Steps 1 + 2, no extra API calls)
    report = verify_citations_sync(result.output, store)
    print(report.text_with_verified_citations)

asyncio.run(main())
```

### Citation format

The LLM is instructed to emit `[REF|key]` tags immediately after any claim derived from a source:

```
The report states revenue grew 12% YoY. [REF|d_1]
```

Multi-source claims use pipe-separated keys:

```
Both documents confirm the finding. [REF|d_1|d_2]
```

---

## Multi-agent usage

Share the same `ProvenanceCapability` store across a coordinator and its subagents:

```python
from pydantic_ai import Agent
from pydantic_ai_provenance import ProvenanceCapability

research_cap = ProvenanceCapability(agent_name="researcher", source_tools=["fetch_url"])
coord_cap    = ProvenanceCapability(agent_name="coordinator")

research_agent = Agent("anthropic:claude-haiku-4-5-20251001", capabilities=[research_cap])
coord_agent    = Agent("anthropic:claude-sonnet-4-6",          capabilities=[coord_cap])

@research_agent.tool_plain
def fetch_url(url: str) -> str: ...

@coord_agent.tool
async def delegate(ctx, topic: str) -> str:
    result = await research_agent.run(f"Research: {topic}", usage=ctx.usage)
    return result.output

async def main():
    result = await coord_agent.run("Summarise pydantic-ai.")
    # Both agents share the same store automatically via contextvars
    store = coord_cap.store
    print(to_mermaid(store))
```

---

## API reference

### Core

| Symbol | Description |
|---|---|
| `ProvenanceCapability` | pydantic-ai `AbstractCapability` that hooks into agent lifecycle |
| `ProvenanceStore` | Central registry: graph + citation key → node mapping |

### Graph primitives

| Symbol | Description |
|---|---|
| `ProvenanceGraph` | DAG container with path traversal helpers |
| `ProvenanceNode` | Single execution step (id, type, label, data, timestamp) |
| `ProvenanceEdge` | Directed edge with optional label |
| `NodeType` | Enum: `INPUT`, `DATA_READ`, `TOOL_CALL`, `TOOL_RESULT`, `MODEL_REQUEST`, `MODEL_RESPONSE`, `AGENT_RUN`, `FINAL_OUTPUT` |

### Attribution

| Symbol | Description |
|---|---|
| `attribute_output(store, output_node_id=None)` | Full path attribution for one `FINAL_OUTPUT` node |
| `attribute_all_outputs(store)` | Attribution for every `FINAL_OUTPUT` |
| `AttributionResult` | `.sources`, `.paths`, `.summary()` |
| `AttributionPath` | Single source-to-output path with `.hop_count` |

### Citations

| Symbol | Description |
|---|---|
| `parse_citations(text)` | Extract all `[REF|…]` tags → `list[CitationRef]` |
| `citation_tag_spans(text)` | Same but with `(start, end, CitationRef)` positions |
| `strip_inline_citation_tags(text)` | Remove all `[REF|…]` tags |
| `strip_inline_citation_tags_preserve_leading_ref_header(text)` | Strip body tags but keep an opening block header |

### Verification

| Symbol | Description |
|---|---|
| `verify_citations_sync(text, store)` | Steps 1 (key sanitisation) + 2 (TF-IDF overlap) |
| `strip_unresolvable_citation_keys(text, store)` | Step 1 only: remove keys not in the store |
| `claim_source_tfidf_cosine(claim, source)` | Max cosine similarity over sliding source windows |
| `entailment_agent(model)` | Build a pydantic-ai Step 3 entailment judge |
| `refine_claim_source_similarities(records)` | Narrow results by top-N and min-score filters |
| `CitationVerificationReport` | `.original_text`, `.text_with_verified_citations`, `.claim_source_similarities` |

### Visualisation

| Symbol | Description |
|---|---|
| `to_mermaid(store)` | Mermaid flowchart string |
| `to_dot(store)` | GraphViz DOT string |
| `to_json(store)` | `dict` with `nodes` and `edges` lists |
| `to_json_str(store, indent=2)` | JSON string |

---

## Running the examples

```bash
# Offline verification only (no API keys required)
uv run python example.py --verify-only

# Full example with Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://... AZURE_OPENAI_API_KEY=... uv run python example.py

# Full example with Anthropic
ANTHROPIC_API_KEY=... uv run python example.py
```

---

## Development

```bash
git clone https://github.com/sumitdugar/pydantic-ai-provenance.git
cd pydantic-ai-provenance
uv sync --extra dev
uv run pytest
uv run ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributing guide.

---

## License

[MIT](LICENSE)
