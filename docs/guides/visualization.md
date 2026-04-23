# Visualization

The provenance graph can be exported as an interactive HTML page, a Mermaid flowchart, a GraphViz DOT file, or raw JSON.

---

## Interactive HTML

`store.to_html()` returns a self-contained HTML page with an interactive Cytoscape.js graph. Click any node to see its full metadata in the sidebar.

```python
# Open immediately in the default browser
store.open_in_browser()

# Or save to a file
html = store.to_html(title="My Run")
with open("provenance.html", "w") as f:
    f.write(html)
```

The page loads Cytoscape.js and the dagre layout plugin from CDN, so a network connection is required at render time. The generated file is otherwise completely self-contained and works offline once loaded.

### Features

- Left-to-right dagre layout matching the natural data-flow direction
- Color-coded nodes by type (same palette as Mermaid / DOT)
- Distinct node shapes per type: barrel for data sources, diamond for model steps, ellipse for final outputs, etc.
- Sidebar legend showing only the node types present in the graph
- Click any node to inspect its label, type, agent, run ID, timestamp, and any extra metadata

---

## Mermaid

`store.to_mermaid()` returns a [Mermaid](https://mermaid.js.org/) flowchart string.

```python
print(store.to_mermaid())
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

`store.to_dot()` returns a [GraphViz](https://graphviz.org/) DOT string.

```python
dot_src = store.to_dot(graph_name="my_run")
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

`store.to_json()` returns a plain Python `dict`; `store.to_json_str()` serialises it to a JSON string.

```python
data = store.to_json()
print(data["nodes"])   # list of node dicts
print(data["edges"])   # list of edge dicts

# Pretty-print
print(store.to_json_str(indent=2))
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
