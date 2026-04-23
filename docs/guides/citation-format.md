# Citation Format

pydantic-ai-provenance uses a compact inline citation format: `[REF|key]`.

---

## Tag syntax

| Pattern | Meaning |
|---|---|
| `[REF|d_1]` | Claim supported by data source `d_1` |
| `[REF|d_1\|d_2]` | Claim supported by both `d_1` and `d_2` |
| `[REF|a_1]` | Claim derived from subagent output `a_1` |

Tags are placed **immediately after** the sentence or phrase they support:

```
Revenue grew 12% year-over-year. [REF|d_1]
Both reports confirm the finding. [REF|d_1|d_2]
```

---

## Key prefixes

| Prefix | Source type | Example |
|---|---|---|
| `d_` | `DATA_READ` node — raw data returned by a source tool | `d_1`, `d_2` |
| `a_` | `FINAL_OUTPUT` node — output from a subagent | `a_1`, `a_2` |

Keys are assigned sequentially from a shared counter on the `ProvenanceStore`. The same counter is used for both `d_` and `a_` keys, so the sequence is globally unique within a session.

---

## How keys are injected

When a source tool (listed in `source_tools`) returns a result, `ProvenanceCapability` wraps it before it reaches the LLM:

```
[REF|d_1]
<original tool result>
```

The block header `[REF|d_1]` tells the LLM which key to use when citing this source. Any inline `[REF|…]` tags already present in the tool result body are stripped to avoid duplication.

Subagent outputs are wrapped the same way before being returned to the parent agent.

---

## Citation instructions

By default (`inject_citation_instructions=True`), the capability injects formatting instructions into the agent's system prompt automatically. This tells the LLM:

- When to cite (facts, statistics, findings, paraphrased arguments).
- When not to cite (reasoning, connective language, common knowledge).
- To prefer original source keys (`d_*`) over subagent keys (`a_*`) when both are available.
- Never to invent keys or cite on repeat mentions.

To manage the prompt yourself, set `inject_citation_instructions=False` and include the instructions in your own system prompt.

---

## Resolving a key

```python
key = "d_1"
node_id = store.resolve_citation(key)
node = store.graph.nodes[node_id]
print(node.data.get("file_path"))
```

Or use `citation_summary()` for a human-readable overview of all registered keys:

```python
for key, meta in store.citation_summary().items():
    print(key, "→", meta["label"], meta.get("file_path", ""))
```
