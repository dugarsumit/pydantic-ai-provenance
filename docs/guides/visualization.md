# Visualization

The provenance graph can be exported in three formats: Mermaid, GraphViz DOT, and JSON.

---

## Mermaid

`to_mermaid(store)` returns a [Mermaid](https://mermaid.js.org/) flowchart string.

```python
from pydantic_ai_provenance import to_mermaid

print(to_mermaid(store))
```

Paste the output into [mermaid.live](https://mermaid.live) or embed it in a GitHub Markdown file:

````markdown
```mermaid
flowchart LR
    ...
```
````

### Node colours

| Node type | Colour |
|---|---|
| `INPUT` | Blue |
| `DATA_READ` | Green |
| `TOOL_CALL` | Orange |
| `TOOL_RESULT` | Amber |
| `MODEL_REQUEST` | Purple |
| `MODEL_RESPONSE` | Violet |
| `AGENT_RUN` | Dark slate |
| `FINAL_OUTPUT` | Red |

---

## GraphViz DOT

`to_dot(store)` returns a [GraphViz](https://graphviz.org/) DOT string.

```python
from pydantic_ai_provenance import to_dot

dot_src = to_dot(store, graph_name="my_run")
print(dot_src)
```

Render it with the `dot` CLI:

```bash
echo "$DOT_SRC" | dot -Tsvg -o provenance.svg
```

Or use the Python `graphviz` package:

```python
import graphviz
g = graphviz.Source(dot_src)
g.render("provenance", format="svg", view=True)
```

---

## JSON

`to_json(store)` returns a plain Python `dict`; `to_json_str(store)` serialises it to a JSON string.

```python
from pydantic_ai_provenance import to_json, to_json_str
import json

data = to_json(store)
print(data["nodes"])   # list of node dicts
print(data["edges"])   # list of edge dicts

# Pretty-print
print(to_json_str(store, indent=2))
```

### Schema

**Node**

```json
{
  "id": "...",
  "type": "data_read",
  "label": "[source] Tool: read_file",
  "agent_name": "summariser",
  "run_id": "...",
  "timestamp": "2024-01-01T00:00:00+00:00",
  "data": { "file_path": "report.txt" }
}
```

**Edge**

```json
{
  "source": "<node_id>",
  "target": "<node_id>",
  "label": "cited_in"
}
```

---

## Working directly with the graph

If none of the built-in formats suit you, the full graph is available as `store.graph`:

```python
graph = store.graph

for node in graph.nodes.values():
    print(node.id, node.type, node.label)

for edge in graph.edges:
    print(edge.source_id, "→", edge.target_id, f"({edge.label})")

# Ancestor traversal
ancestors = graph.ancestors(some_node_id)

# All source-to-output paths
paths = graph.all_paths_to_sources(output_node_id)
```
