from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .citations import citation_tag_spans
from .graph import NodeType, ProvenanceGraph
from .store import ProvenanceStore

try:
    from pydantic_ai import Agent as _PydanticAgent
except ImportError:  # pragma: no cover
    _PydanticAgent = None


@dataclass
class CitationKeyFilterResult:
    """Records which keys were kept or removed from a single [REF|...] tag."""

    raw_tag: str
    tag_start: int
    tag_end: int
    original_keys: list[str]
    removed_keys: list[str]
    retained_keys: list[str]


@dataclass
class ClaimSourceSimilarity:
    """Similarity between a claim's context and one resolved source segment."""

    method: str
    claim_key: str
    source_keys: list[str]
    scores: list[float]
    claim_excerpt: str
    source_excerpts: list[str]
    source_file_paths: list[str]
    raw_tag: str
    verified_tag: str
    tag_start: int | None = None
    tag_end: int | None = None


@dataclass
class EntailmentRecord:
    """Step 3 semantic support (optional)."""

    citation_key: str
    probability: float
    rationale: str = ""


@dataclass
class CitationVerificationReport:
    """Aggregated verification for a single user/model string."""

    original_text: str
    text_with_verified_citations: str
    claim_source_similarities: list[ClaimSourceSimilarity] = field(default_factory=list)


def _strip_ref_block_header(text: str) -> str:
    """Remove a leading provenance header line from stored tool output if present."""
    if not text:
        return text
    lines = text.splitlines()
    if not lines:
        return text
    first = lines[0].strip()
    if first.startswith("[REF|") and first.endswith("]"):
        return "\n".join(lines[1:]).lstrip("\n")
    return text


def _tool_result_raw_text(store: ProvenanceStore, source_call_node_id: str) -> str | None:
    """Return the raw tool result string for a DATA_READ call node."""
    graph = store.graph
    for succ in graph.successors(source_call_node_id):
        if succ.type == NodeType.TOOL_RESULT:
            raw = succ.data.get("result")
            raw = _strip_ref_block_header(raw)
            return str(raw) if raw is not None else ""
    return None


def _citation_resolved_text(store: ProvenanceStore, key: str) -> str | None:
    """Text blob associated with a citation key (for overlap / entailment checks)."""
    node_id = store.resolve_citation(key)
    if node_id is None:
        return None
    node = store.graph.nodes.get(node_id)
    if node is None:
        return None
    if node.type == NodeType.DATA_READ:
        return _tool_result_raw_text(store, node_id)
    if node.type == NodeType.FINAL_OUTPUT:
        out = node.data.get("output")
        return str(out) if out is not None else ""
    return None


def _cited_in_source_node_ids(graph: ProvenanceGraph, target_node_id: str) -> list[str]:
    """Return source node ids that have a ``cited_in`` edge into ``target_node_id``."""
    return [
        e.source_id
        for e in graph.edges
        if e.target_id == target_node_id and e.label == "cited_in"
    ]


def _gather_source_segments(store: ProvenanceStore, citation_key: str) -> list[tuple[str, str, str]]:
    """Collect (source_citation_key, text, file_path) pairs to compare against a claim.

    For d_* citation_keys, returns the single resolved text. 
    For a_* citation_keys, also includes text from direct upstream sources (e.g.
    underlying data keys) when those edges exist, so claim can be checked against primary
    documents as well as the wrapped agent answer.
    """
    node_id = store.resolve_citation(citation_key)
    if node_id is None:
        return []
    node = store.graph.nodes.get(node_id)
    if node is None:
        return []

    segment_keys = {citation_key}

    if node.type == NodeType.FINAL_OUTPUT:
        for source_node_id in _cited_in_source_node_ids(store.graph, node_id):
            source_key = store.citation_key_for_node(source_node_id)
            if source_key:
                segment_keys.add(source_key)

    segments: list[tuple[str, str, str]] = []
    for key in segment_keys:
        text = _citation_resolved_text(store, key)
        segments.append((key, text, node.data.get("file_path")))

    return segments


def _normalize_for_overlap(s: str) -> str:
    whitespace_re = re.compile(r"\s+")
    return whitespace_re.sub(" ", s.strip().lower())


def _chunk_source_windows(
    source_norm: str,
    *,
    window_chars: int,
    stride_chars: int,
    max_chunks: int,
) -> list[str]:
    if not source_norm.strip():
        return []
    if len(source_norm) <= window_chars:
        return [source_norm]
    chunks: list[str] = []
    end = len(source_norm) - window_chars
    i = 0
    while i <= end and len(chunks) < max_chunks:
        chunks.append(source_norm[i : i + window_chars])
        i += stride_chars
    tail = source_norm[-window_chars:]
    if tail != chunks[-1] and len(chunks) < max_chunks:
        chunks.append(tail)
    return chunks


def claim_source_tfidf_cosine(
    claim_text: str,
    source_text: str,
    *,
    max_source_chars: int = 96_000,
    chunk_chars: int = 1_200,
    chunk_stride: int = 600,
    max_chunks: int = 400,
) -> float:
    """Computes the maximum TF-IDF cosine similarity between a "claim" text and sliding window chunks of a "source" text. It does this by:
    - Creating a TF-IDF matrix over the claim and all source chunks.
    - Calculating cosine similarity between the claim vector and each source chunk vector.
    - Returning the highest similarity score (with edge cases handled for empty/NaN results, and the result clipped to [0, 1]).
    - This measures how lexically similar the claim is to any chunk of the source.
    """
    claim_text = _normalize_for_overlap(claim_text)
    source_text = _normalize_for_overlap(source_text[:max_source_chars])
    if not claim_text.strip() or not source_text.strip():
        return 0.0
    
    # Split source text into chunks for comparison
    chunks = _chunk_source_windows(
        source_text, window_chars=chunk_chars, stride_chars=chunk_stride, max_chunks=max_chunks
    )
    if not chunks:
        return 0.0
    
    docs = [claim_text, *chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)[a-z0-9]+",
        min_df=1,
    )
    matrix = vectorizer.fit_transform(docs)
    if matrix.shape[1] == 0:
        return 0.0
    
    sims = cosine_similarity(matrix[0:1], matrix[1:])
    m = float(sims.max())
    if m != m:  # NaN if zero vectors
        return 0.0
    return min(1.0, max(0.0, m))


def _split_sentences_simple(text: str) -> list[str]:
    """Split on ``.!?`` + whitespace; last segment may lack terminal punctuation (text up to the citation)."""
    # Sentence boundaries for claim context: . ! ? followed by whitespace (simple, no abbreviation handling).
    sentence_split_re = re.compile(r"(?<=[.!?])\s+")
    if not text or not text.strip():
        return []
    parts = sentence_split_re.split(text.strip())
    return [p for p in parts if p]


def _last_paragraph(text: str) -> str:
    """Text after the last blank-line paragraph break, or the whole string if none."""
    # Blank-line paragraph breaks (do not treat single newlines as paragraph boundaries).
    paragraph_split_re = re.compile(r"\n\s*\n")
    t = text.strip()
    if not t:
        return ""
    blocks = paragraph_split_re.split(t)
    return blocks[-1].strip() if blocks else t


def context_before_span(
    text: str,
    start: int,
    *,
    max_chars: int = 720,
    max_sentences: int = 1,
) -> str:
    """
    Text immediately before a citation span for lexical overlap: up to the last ``max_sentences``
    sentences (default 1) in the last blank-line paragraph of a ``max_chars`` tail of ``text[:start]``,
    without crossing a prior ``[REF|…]`` tag in that window.

    Uses a simple ``.!?`` + whitespace split so a citation at the end of a long paragraph yields a
    short local claim instead of the full paragraph (or prior paragraphs). Pass ``max_sentences=2`` for
    a little more surrounding context.
    """
    prefix = text[:start]
    if len(prefix) > max_chars:
        prefix_window = prefix[-max_chars:]
    else:
        prefix_window = prefix

    prev_ref_match = None
    for match in re.finditer(r"\[REF\|[^\]]+\]", prefix_window):
        prev_ref_match = match

    if prev_ref_match:
        body = prefix_window[prev_ref_match.end() :]
    else:
        body = prefix_window

    body = body.strip()
    if not body:
        return ""

    scoped = _last_paragraph(body)
    if not scoped:
        scoped = body

    sentences = _split_sentences_simple(scoped)
    if not sentences:
        return scoped
    tail = sentences[-max_sentences:] if len(sentences) > max_sentences else sentences
    return " ".join(s.strip() for s in tail).strip()


class EntailmentJudge(Protocol):
    async def score(self, *, source_excerpt: str, claim: str) -> tuple[float, str]: ...


class EntailmentJudgment(BaseModel):
    """Structured output for the default entailment :class:`~pydantic_ai.agent.Agent`."""

    probability_support: float = Field(
        ge=0.0,
        le=1.0,
        description="How likely the premise supports the hypothesis (0-1).",
    )
    rationale: str = Field(default="", description="One short sentence.")


_DEFAULT_ENTAILMENT_INSTRUCTIONS = (
    "You judge whether a short premise (excerpt from a cited source) supports a "
    "hypothesis (the author's claim next to the citation). "
    "Reply using the structured schema only. "
    "Use probability_support near 1 only when the premise clearly entails or strongly "
    "supports the hypothesis; use low values when support is weak or absent."
)


def entailment_agent(
    model: Any,
    *,
    instructions: str | None = None,
) -> Any:
    """Build a small pydantic-ai agent for Step 3 (caller supplies the model)."""
    if _PydanticAgent is None:  # pragma: no cover
        raise ImportError("pydantic_ai is required for entailment_agent()")
    return _PydanticAgent(
        model,
        output_type=EntailmentJudgment,
        instructions=instructions or _DEFAULT_ENTAILMENT_INSTRUCTIONS,
    )


async def _score_with_pydantic_agent(
    agent: Any,
    *,
    source_excerpt: str,
    claim: str,
    max_premise_chars: int = 6_000,
    max_hypothesis_chars: int = 1_200,
) -> EntailmentJudgment:
    premise = source_excerpt[:max_premise_chars]
    hyp = claim[:max_hypothesis_chars]
    prompt = (
        "Premise (source excerpt):\n"
        f"{premise}\n\n"
        "Hypothesis (cited claim):\n"
        f"{hyp}"
    )
    result = await agent.run(prompt)
    return result.output


def strip_unresolvable_citation_keys(
    text: str, store: ProvenanceStore
) -> tuple[str, list[CitationKeyFilterResult]]:
    """Remove citation keys that don't resolve in the store; drop tags left empty."""
    citation_tags: list[CitationKeyFilterResult] = []
    sanitized_text = text

    for tag_start, tag_end, citation_tag in reversed(citation_tag_spans(text)):
        keys = citation_tag.refs
        kept = [k for k in keys if store.resolve_citation(k) is not None]

        if kept == keys:
            continue

        removed = list(set(keys) - set(kept))
   
        replacement_tag = "" if not kept else "[REF|" + "|".join(kept) + "]"
        sanitized_text = sanitized_text[:tag_start] + replacement_tag + sanitized_text[tag_end:]

        citation_tags.append(
            CitationKeyFilterResult(
                raw_tag=citation_tag.raw,
                tag_start=tag_start,
                tag_end=tag_end,
                original_keys=list(keys),
                removed_keys=removed,
                retained_keys=list(kept),
            )
        )

    citation_tags.reverse()
    return sanitized_text, citation_tags


def compare_claim_to_sources_tfidf(
    text: str,
    store: ProvenanceStore,
    *,
    claim_max_chars: int = 720,
    max_source_chars: int = 96_000,
    chunk_chars: int = 1_200,
    chunk_stride: int = 600,
    max_chunks: int = 400,
) -> list[ClaimSourceSimilarity]:
    """Step 2: TF-IDF cosine between claim context and each resolved source segment for that citation."""
    results: list[ClaimSourceSimilarity] = []
    for tag_start, tag_end, citation_tag in citation_tag_spans(text):
        claim_text = context_before_span(text, tag_start, max_chars=claim_max_chars)
        for claim_key in citation_tag.refs:
            source_segments = _gather_source_segments(store, claim_key)
            if not source_segments:
                continue

            scores = []
            source_keys = []
            source_excerpts = []
            source_file_paths = []
            for source_citation_key, source_text, source_file_path in source_segments:
                score = claim_source_tfidf_cosine(
                    claim_text=claim_text,
                    source_text=source_text,
                    max_source_chars=max_source_chars,
                    chunk_chars=chunk_chars,
                    chunk_stride=chunk_stride,
                    max_chunks=max_chunks,
                )
                source_keys.append(source_citation_key)
                scores.append(float(score))
                source_excerpts.append(source_text)
                source_file_paths.append(source_file_path)
            results.append(
                ClaimSourceSimilarity(
                    method="tfidf",
                    claim_key=claim_key,
                    source_keys=source_keys,
                    scores=scores,
                    claim_excerpt=claim_text,
                    source_excerpts=source_excerpts,
                    source_file_paths=source_file_paths,
                    raw_tag=citation_tag.raw,
                    verified_tag="[REF|" + "|".join(source_keys) + "]",
                    tag_start=tag_start,
                    tag_end=tag_end,
                )
            )
    return results


def _keep_top_citation_keys_per_tag(
    records: list[ClaimSourceSimilarity],
    top_n: int,
) -> list[ClaimSourceSimilarity]:
    """For multi-key citation tags, keep only the top-N keys ranked by best cosine score.

    Returns a tuple: (kept_records, removed_records).
    """

    for record in records:
        if len(record.source_keys) <= top_n:
            continue

        multi_key_records = []
        for i, key in enumerate(record.source_keys):
            multi_key_records.append((record.scores[i], key, record.source_excerpts[i], record.source_file_paths[i]))
        
        # Sort multi_key_records by score descending
        multi_key_records.sort(key=lambda x: (-x[0], x[1]))

        kept_keys = [x for x in multi_key_records[:top_n]]

        # update record
        record.scores = [x[0] for x in kept_keys]
        record.source_keys = [x[1] for x in kept_keys]
        record.source_excerpts = [x[2] for x in kept_keys]
        record.source_file_paths = [x[3] for x in kept_keys]

        # update record.verified_tag with the kept keys
        record.verified_tag = "[REF|" + "|".join(record.source_keys) + "]"

    return records


def _drop_weak_sources(
    records: list[ClaimSourceSimilarity],
    min_score: float,
) -> list[ClaimSourceSimilarity]:
    """Drop low-scoring citation keys."""

    for record in records:
        strong_keys = []
        strong_scores = []
        strong_source_excerpts = []
        strong_source_file_paths = []
        for i, _ in enumerate(record.source_keys):
            if record.scores[i] < min_score:
                continue
            strong_keys.append(record.source_keys[i])
            strong_scores.append(record.scores[i])
            strong_source_excerpts.append(record.source_excerpts[i])
            strong_source_file_paths.append(record.source_file_paths[i])
            
        record.source_keys = strong_keys
        record.scores = strong_scores
        record.source_excerpts = strong_source_excerpts
        record.source_file_paths = strong_source_file_paths
        record.verified_tag = "[REF|" + "|".join(strong_keys) + "]" if strong_keys else ""

    return records


def refine_claim_source_similarities(
    records: list[ClaimSourceSimilarity],
    *,
    max_top_n_keys_per_tag: int = 2,
    min_score_for_shared_source: float = 0.3,
) -> list[ClaimSourceSimilarity]:
    """Narrow down similarity records for cleaner provenance signals.

    Two optional passes:
      - For multi-key citation tags ([REF|k1|k2|...]), keep only the top-N
        citation keys ranked by their best cosine score. Single-key tags are
        unchanged.
      - If the same source supports claims at two or more distinct sites,
        drop records below the minimum cosine threshold.
    """
    result = list(records)

    if min_score_for_shared_source is not None:
        result = _drop_weak_sources(result, min_score_for_shared_source)

    if max_top_n_keys_per_tag is not None and max_top_n_keys_per_tag > 0:
        result = _keep_top_citation_keys_per_tag(result, max_top_n_keys_per_tag)

    return result


# async def verify_citations(
#     text: str,
#     store: ProvenanceStore,
#     *,
#     context_max_chars: int = 720,
#     entailment_judge: Any | None = None,
#     max_entailment_source_chars: int = 6_000,
# ) -> CitationVerificationReport:
#     """Run Step 1 (always), Step 2 (always), and Step 3 when ``entailment_judge`` is set.

#     ``entailment_judge`` may be an :func:`entailment_agent` instance or any object
#     with ``async def score(*, source_excerpt, claim) -> tuple[float, str]``.
#     """
#     sanitized, sanitize_records = strip_unresolvable_citation_keys(text, store)
#     lexical = compare_claim_to_sources_tfidf(sanitized, store, context_max_chars=context_max_chars)
#     entailment: list[EntailmentRecord] = []

#     if entailment_judge is not None:
#         judge = entailment_judge
#         use_pydantic_agent = _PydanticAgent is not None and isinstance(judge, _PydanticAgent)
#         best_by_citation: dict[str, ClaimSourceSimilarity] = {}
#         for rec in lexical:
#             prev = best_by_citation.get(rec.citation_key)
#             if prev is None or rec.tfidf_cosine > prev.tfidf_cosine:
#                 best_by_citation[rec.citation_key] = rec
#         for rec in best_by_citation.values():
#             blob = _citation_resolved_text(store, rec.source_key) or _citation_resolved_text(
#                 store, rec.citation_key
#             )
#             if not blob:
#                 for sk, _ in expand_citation_texts_for_overlap(store, rec.citation_key):
#                     blob = _citation_resolved_text(store, sk)
#                     if blob:
#                         break
#             if not blob:
#                 continue
#             excerpt = blob[:max_entailment_source_chars]
#             if use_pydantic_agent:
#                 judgment = await _score_with_pydantic_agent(
#                     judge,
#                     source_excerpt=excerpt,
#                     claim=rec.claim_excerpt,
#                 )
#                 entailment.append(
#                     EntailmentRecord(
#                         citation_key=rec.citation_key,
#                         probability=judgment.probability_support,
#                         rationale=judgment.rationale,
#                     )
#                 )
#             else:
#                 protocol_judge = cast(EntailmentJudge, judge)
#                 prob, rationale = await protocol_judge.score(
#                     source_excerpt=excerpt,
#                     claim=rec.claim_excerpt,
#                 )
#                 prob = max(0.0, min(1.0, float(prob)))
#                 entailment.append(
#                     EntailmentRecord(
#                         citation_key=rec.citation_key,
#                         probability=prob,
#                         rationale=rationale,
#                     )
#                 )

#     return CitationVerificationReport(
#         original_text=text,
#         sanitized_text=sanitized,
#         sanitize_records=sanitize_records,
#         lexical=lexical,
#         entailment=entailment,
#     )


def _remove_bad_citation_keys(text: str, verified_claim_source_similarities: list[ClaimSourceSimilarity]) -> str:
    """Remove citation keys that failed verification."""
    # sort verified_claim_source_similarities by tag_end descending
    verified_claim_source_similarities.sort(key=lambda x: x.tag_end, reverse=True)

    for record in verified_claim_source_similarities:
        text = text[:record.tag_start] + record.verified_tag + text[record.tag_end:]
    return text


def verify_citations_sync(
    text: str,
    store: ProvenanceStore,
    *,
    context_max_chars: int = 720,
) -> CitationVerificationReport:
    """Steps 1-2 only (no LLM). Async :func:`verify_citations` adds Step 3."""
    sanitized_text, _ = strip_unresolvable_citation_keys(text, store)
    claim_source_similarities = compare_claim_to_sources_tfidf(
        sanitized_text, store, claim_max_chars=context_max_chars
    )
    verified_claim_source_similarities = refine_claim_source_similarities(claim_source_similarities)
    text_with_verified_citations = _remove_bad_citation_keys(
        sanitized_text, verified_claim_source_similarities
    )
    return CitationVerificationReport(
        original_text=text,
        text_with_verified_citations=text_with_verified_citations,
        claim_source_similarities=claim_source_similarities,
    )
