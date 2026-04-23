# pydantic-ai-provenance

**Provenance tracking and citation verification for [pydantic-ai](https://ai.pydantic.dev/) agents.**

Attach `ProvenanceCapability` to any pydantic-ai agent and get a complete, auditable record of where every claim in the model's output came from.

---

## What it does

```
┌──────────┐   tool call    ┌───────────┐   [REF|d_1]   ┌──────────────┐
│  Agent   │ ─────────────▶ │  Source   │ ─────────────▶ │  LLM output  │
│          │                │  (file,   │               │              │
│          │ ◀───────────── │   API…)   │               │  "Revenue    │
│          │  wrapped result│           │               │  grew 12%.   │
└──────────┘                └───────────┘               │  [REF|d_1]"  │
                                                         └──────────────┘
                                                                │
                                                  provenance.verify()
                                                                │
                                                     ✓ TF-IDF overlap check
```

- **Execution DAG** — every tool call, model request, and response captured as a typed graph.
- **Citation keys** — source tool results are automatically tagged `[REF|d_1]`, `[REF|d_2]` … so the LLM can cite them inline.
- **Multi-agent attribution** — subagent outputs propagate through a shared store, giving transitive provenance across any number of agent hops.
- **Citation verification** — TF-IDF cosine overlap (Step 2) and optional LLM entailment (Step 3) validate every `[REF|…]` tag.
- **Visualisation** — export the provenance graph as Mermaid, GraphViz DOT, or JSON.

---

## Install

```bash
pip install pydantic-ai-provenance
```

```bash
uv add pydantic-ai-provenance
```

---

## 30-second example

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai_provenance.capability import ProvenanceCapability
from pydantic_ai_provenance.attribution import attribute_output

provenance = ProvenanceCapability(
    agent_name="summariser",
    source_tools=["read_file"],
)

agent = Agent(
    "anthropic:claude-sonnet-4-6",
    capabilities=[provenance],
)

@agent.tool_plain
def read_file(path: str) -> str:
    return open(path).read()

async def main():
    result = await agent.run("Summarise report.txt.")
    store = provenance.store

    print(attribute_output(store).summary())   # path-level attribution
    print(store.to_mermaid())                  # copy into mermaid.live
    report = await provenance.verify(result.output)
    print(report.text_with_verified_citations) # citations validated

asyncio.run(main())
```

---

## Next steps

- [Getting Started](getting-started.md) — full setup walkthrough
- [Citation Format](guides/citation-format.md) — how `[REF|…]` tags work
- [Citation Verification](guides/verification.md) — Steps 1, 2, and 3 explained
- [Multi-Agent Usage](guides/multi-agent.md) — coordinator + subagent patterns
- [Visualization](guides/visualization.md) — Mermaid, DOT, and JSON exports
- [API Reference](api-reference.md) — complete symbol listing
