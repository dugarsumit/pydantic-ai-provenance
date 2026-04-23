# Getting Started

## Installation

```bash
pip install pydantic-ai-provenance
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pydantic-ai-provenance
```

**Requirements:** Python ≥ 3.10 and pydantic-ai ≥ 1.80.

---

## Basic setup

`ProvenanceCapability` is a pydantic-ai `AbstractCapability`. Pass it to any agent via the `capabilities` list — no other changes to your agent code are needed.

```python
from pydantic_ai import Agent
from pydantic_ai_provenance import ProvenanceCapability

provenance = ProvenanceCapability(
    agent_name="my-agent",   # label used in the graph
    source_tools=["read_file", "fetch_url"],  # tools that return raw data
)

agent = Agent(
    "anthropic:claude-sonnet-4-6",
    capabilities=[provenance],
)
```

### `source_tools`

List the names of tools whose return values are **raw data sources** (file readers, API fetchers, database queries, etc.). Each invocation of a source tool gets a unique citation key (`d_1`, `d_2`, …) automatically injected into the result seen by the LLM.

Tools not in `source_tools` are still tracked in the graph as `TOOL_CALL` nodes, but their results are not wrapped with a citation key.

---

## Running the agent

```python
import asyncio

@agent.tool_plain
def read_file(path: str) -> str:
    return open(path).read()

async def main():
    result = await agent.run("Read report.txt and summarise it.")
    store = provenance.store   # access the store after the run

asyncio.run(main())
```

The `store` is available on the `ProvenanceCapability` instance after the run completes.

---

## Inspecting the provenance

### Path-level attribution

```python
from pydantic_ai_provenance import attribute_output

attribution = attribute_output(store)
print(attribution.summary())
```

Example output:

```
Output: Final output: my-agent

Contributing sources (1):
  [data] [source] Tool: read_file

Attribution paths (1):
  1. [source] Tool: read_file → Result: read_file → … → Final output: my-agent
```

### Citation summary

```python
print(store.citation_summary())
# {'d_1': {'type': 'data_read', 'label': '...', 'file_path': 'report.txt', ...}}
```

---

## Verifying citations

```python
report = await provenance.verify(result.output)
print(report.text_with_verified_citations)
```

`verify()` runs Step 1 (key sanitisation) and Step 2 (TF-IDF overlap scoring) and returns a `CitationVerificationReport`. See [Citation Verification](guides/verification.md) for the full three-step pipeline.

---

## Visualising the graph

```python
print(store.to_mermaid())
```

Paste the output into [mermaid.live](https://mermaid.live) to see the full execution DAG.

See [Visualization](guides/visualization.md) for Mermaid, GraphViz DOT, and JSON options.

---

## Running the bundled examples

```bash
# Offline citation verification (no API keys required)
uv run python examples/verify_citations.py

# Single-agent example
ANTHROPIC_API_KEY=... uv run python examples/single_agent.py

# Multi-agent example
ANTHROPIC_API_KEY=... uv run python examples/multi_agent.py
```
