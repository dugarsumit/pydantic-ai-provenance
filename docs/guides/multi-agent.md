# Multi-Agent Usage

pydantic-ai-provenance tracks provenance across multiple agents in the same session via a shared `ProvenanceStore` propagated through Python's `contextvars`.

---

## How it works

When a parent agent calls a tool that internally runs a subagent, the subagent's `ProvenanceCapability` detects the active context and reuses the parent's `ProvenanceStore`. The subagent's final output is registered in that shared store with an `a_*` citation key, and the result is wrapped with `[REF|a_1]` before being returned to the parent's LLM.

No manual wiring is required — the `ContextVar` propagation is handled automatically.

---

## Setup

Create a separate `ProvenanceCapability` for each agent. They will share a store once the first agent's run starts.

```python
from pydantic_ai import Agent
from pydantic_ai_provenance.capability import ProvenanceCapability

research_cap = ProvenanceCapability(
    agent_name="researcher",
    source_tools=["fetch_url"],
)
coordinator_cap = ProvenanceCapability(
    agent_name="coordinator",
)

research_agent = Agent(
    "anthropic:claude-haiku-4-5-20251001",
    capabilities=[research_cap],
)
coordinator_agent = Agent(
    "anthropic:claude-sonnet-4-6",
    capabilities=[coordinator_cap],
)
```

---

## Delegating to the subagent

```python
@research_agent.tool_plain
def fetch_url(url: str) -> str:
    ...  # fetch and return page content

@coordinator_agent.tool
async def delegate_research(ctx, topic: str) -> str:
    result = await research_agent.run(
        f"Research this topic: {topic}",
        usage=ctx.usage,  # share token budget
    )
    return result.output

async def main():
    result = await coordinator_agent.run(
        "Summarise the latest developments in pydantic-ai."
    )

    # Both agents share the same store
    store = coordinator_cap.store

    print(store.to_mermaid())  # full DAG including both agents
```

---

## Attribution across agents

After the run, `attribute_output` traces paths all the way back through the subagent to the original data sources:

```python
from pydantic_ai_provenance.attribution import attribute_output

attribution = attribute_output(coordinator_cap.store)
print(attribution.summary())
```

```
Output: Final output: coordinator

Contributing sources (2):
  [data] [source] Tool: fetch_url
  [data] [source] Tool: fetch_url

Attribution paths (2):
  1. [source] Tool: fetch_url → … → Final output: researcher → … → Final output: coordinator
  2. [source] Tool: fetch_url → … → Final output: researcher → … → Final output: coordinator
```

---

## Citation depth

The citation instructions injected by default tell the LLM to prefer the most specific available key:

- If the subagent result `[REF|a_1]` contains original-source citations `[REF|d_1]`, use `[REF|d_1]` — not `[REF|a_1]`.
- Use `[REF|a_1]` only when no traceable original key exists.
- Never combine both: `[REF|a_1|d_1]` is discouraged.

This keeps citations as close to primary sources as possible throughout the chain.

---

## Shared store access

The shared store is accessible from **either** capability after the run:

```python
store_from_coordinator = coordinator_cap.store
store_from_researcher  = research_cap.store
assert store_from_coordinator is store_from_researcher  # same object
```

!!! warning
    Using the same `ProvenanceCapability` instance for **concurrent** overlapping runs is not safe. Create a fresh instance per concurrent run if needed.

---

## Visualising the shared graph

After the run, the shared store contains nodes from all agents. Open it as an interactive Cytoscape.js graph directly from Python:

```python
store.open_in_browser(title="Multi-Agent Provenance Graph")
```

Or save to a file:

```python
html_path = Path("provenance.html")
html_path.write_text(store.to_html(title="Multi-Agent Provenance Graph"), encoding="utf-8")
```

Nodes with citation keys are visually distinguished — amber border for data sources (`d_*`), blue border for agent outputs (`a_*`) — making it easy to trace which subagent produced which cited result. See [Visualization](../guides/visualization.md) for the full feature list.
