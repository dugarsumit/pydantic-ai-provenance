# Citation Verification

`verify_citations` (or `provenance.verify()`) runs a two-step pipeline that checks every `[REF|…]` tag in the model's output against the provenance store. An optional third step adds LLM-based semantic entailment.

---

## Step 1 — Key sanitisation

`strip_unresolvable_citation_keys(text, store)` removes any citation key that does not resolve in the store (e.g. hallucinated keys the model invented).

- Tags whose **all** keys are invalid are removed entirely.
- Tags with a **mix** of valid and invalid keys are reduced to the valid subset: `[REF|d_1|bad_key]` → `[REF|d_1]`.

```python
from pydantic_ai_provenance.verification import strip_unresolvable_citation_keys

sanitized, records = strip_unresolvable_citation_keys(text, store)
for r in records:
    print(f"Removed {r.removed_keys} from {r.raw_tag}")
```

---

## Step 2 — TF-IDF cosine overlap

`compare_claim_to_sources_tfidf(text, store)` extracts the claim context immediately before each `[REF|…]` tag and computes the maximum cosine similarity between the claim and the cited source using TF-IDF vectors over sliding windows.

**How claim context is extracted:**

- Takes up to 720 characters of text immediately before the tag (controlled by `claim_context_chars`).
- Uses the last blank-line paragraph of that window.
- Applies simple `.!?` sentence splitting and keeps the last sentence.
- Stops at any previous `[REF|…]` tag to avoid mixing claims.

**How source similarity is computed:**

- Source text is normalised (lowercased, whitespace-collapsed).
- Split into overlapping windows (`source_chunk_chars=1200`, `source_chunk_stride=600`) for long documents.
- TF-IDF vectors are built over all windows, and the maximum cosine similarity is returned.

```python
from pydantic_ai_provenance.verification import compare_claim_to_sources_tfidf

similarities = compare_claim_to_sources_tfidf(sanitized_text, store)
for sim in similarities:
    print(sim.claim_key, max(sim.scores))
```

### Refining results

`refine_claim_source_similarities` narrows down the similarity records:

- **Top-N per tag** (`max_top_n_keys_per_tag=2`): for multi-key tags, keep only the best two sources.
- **Min-score filter** (`min_score_for_shared_source=0.3`): drop keys below the threshold.

```python
from pydantic_ai_provenance.verification import refine_claim_source_similarities

refined = refine_claim_source_similarities(similarities, min_score_for_shared_source=0.3)
```

---

## Combined (Steps 1 + 2)

The simplest way to run both steps is via the capability's `verify()` method:

```python
report = await provenance.verify(result.output)

print(report.original_text)
print(report.text_with_verified_citations)  # tags adjusted/removed after verification
print(report.claim_source_similarities)     # per-tag TF-IDF scores
```

Or call `verify_citations` directly with a store:

```python
from pydantic_ai_provenance.verification import verify_citations

report = await verify_citations(result.output, store)
```

---

## Step 3 — LLM entailment (optional)

For higher-confidence checks, `entailment_agent` wraps a pydantic-ai agent that judges whether a source excerpt semantically supports a claim.

```python
from pydantic_ai import Agent
from pydantic_ai_provenance.verification import entailment_agent

judge = entailment_agent("anthropic:claude-haiku-4-5-20251001")

# score() returns (probability_support: float, rationale: str)
prob, rationale = await judge.score(
    source_excerpt="Revenue grew 12% in Q3.",
    claim="The company reported double-digit revenue growth.",
)
```

`EntailmentJudgment.probability_support` is 0–1; values near 1 indicate strong entailment.

!!! note
    Step 3 makes an additional LLM API call per citation. Use it selectively (e.g. only for citations with borderline TF-IDF scores).

---

## Choosing thresholds

| Score range | Interpretation |
|---|---|
| ≥ 0.6 | Strong lexical overlap — claim closely mirrors source wording |
| 0.3 – 0.6 | Moderate overlap — paraphrase or indirect reference |
| < 0.3 | Weak overlap — consider removing the citation |

The default `min_score_for_shared_source=0.3` drops keys below this threshold when a source is cited at multiple sites in the same response.
