"""Citation verification: key sanitisation (Step 1), TF-IDF overlap (Step 2), and
optional LLM entailment (Step 3)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pydantic_ai_provenance.citations import citation_tag_spans
from pydantic_ai_provenance.graph import NodeType, ProvenanceGraph
from pydantic_ai_provenance.store import ProvenanceStore

try:
    from pydantic_ai import Agent as _PydanticAgent
except ImportError:  # pragma: no cover
    _PydanticAgent = None

# Compiled once at import time — used by normalisation and context-extraction helpers.
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class CitationKeyFilterResult:
    """Records which keys were kept or removed from a single [REF|...] tag.

    Produced by :func:`strip_unresolvable_citation_keys` for every tag that had
    at least one key removed.  Tags where all keys resolved cleanly are not
    represented.

    Attributes:
        raw_tag: The original ``[REF|…]`` tag string as it appeared in the text.
        tag_start: Start character offset of the tag in the original text.
        tag_end: End character offset (exclusive) of the tag in the original text.
        original_keys: All keys found in the tag before filtering.
        removed_keys: Keys that could not be resolved in the store and were removed.
        retained_keys: Keys that resolved successfully and were kept.
    """

    raw_tag: str
    tag_start: int
    tag_end: int
    original_keys: list[str]
    removed_keys: list[str]
    retained_keys: list[str]


@dataclass
class ClaimSourceSimilarity:
    """Similarity between a claim's context and one resolved source segment.

    One record is produced per ``(citation_tag, claim_key)`` pair found in a
    text during Step 2.  When a single tag contains multiple keys (e.g.
    ``[REF|d_1|d_2]``), one record is created per key.

    Attributes:
        method: Algorithm used (``"tfidf"`` for Step 2, custom label for Step 3).
        claim_key: The citation key from the original tag being scored.
        source_keys: Keys of the source segments actually compared (may include
            upstream ``cited_in`` sources for ``a_*`` keys).
        scores: Cosine similarity score for each entry in ``source_keys``.
        claim_excerpt: The extracted claim text used for comparison.
        source_excerpts: Raw text blobs of the corresponding source segments.
        source_file_paths: File path metadata for each source segment (may be
            ``None`` if unavailable).
        raw_tag: The original ``[REF|…]`` tag string before verification.
        verified_tag: The potentially modified tag after filtering weak keys
            (may be empty string if all keys were dropped).
        tag_start: Start character offset of the tag in the text, or ``None``
            if positional information is not available.
        tag_end: End character offset (exclusive) of the tag, or ``None``.
    """

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
    """Step 3 semantic support result for one citation key.

    Attributes:
        citation_key: The citation key that was evaluated (e.g. ``"d_1"``).
        probability: Estimated probability in ``[0, 1]`` that the cited source
            supports the associated claim.  Values near 1 indicate strong support.
        rationale: Optional one-sentence explanation produced by the entailment
            judge.
    """

    citation_key: str
    probability: float
    rationale: str = ""


@dataclass
class CitationVerificationReport:
    """Aggregated verification result for a single text string.

    Attributes:
        original_text: The input text before any modification.
        text_with_verified_citations: The text after Steps 1 and 2 have removed
            unresolvable keys and replaced tags with their verified forms.
        claim_source_similarities: One :class:`ClaimSourceSimilarity` record per
            ``(tag, key)`` pair scored during Step 2.
    """

    original_text: str
    text_with_verified_citations: str
    claim_source_similarities: list[ClaimSourceSimilarity] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers — source text retrieval
# ---------------------------------------------------------------------------


def _strip_ref_block_header(text: str) -> str:
    """Remove a leading ``[REF|<key>]`` block header from stored tool output if present."""
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
    """Return the text blob associated with a citation key, for overlap / entailment checks.

    For ``DATA_READ`` nodes the raw tool-result text is returned (with the
    ``[REF|…]`` block header stripped).  For ``FINAL_OUTPUT`` nodes the stored
    ``output`` value is returned.  Returns ``None`` if the key cannot be resolved
    or maps to an unsupported node type.

    Args:
        store: The :class:`~.store.ProvenanceStore` holding the registry and graph.
        key: A citation key such as ``"d_1"`` or ``"a_2"``.

    Returns:
        The raw text string, an empty string when the value is present but blank,
        or ``None`` when the key or node cannot be found.
    """
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
    """Return the IDs of nodes that have a ``cited_in`` edge pointing to ``target_node_id``."""
    return [e.source_id for e in graph.edges if e.target_id == target_node_id and e.label == "cited_in"]


def _gather_source_segments(store: ProvenanceStore, citation_key: str) -> list[tuple[str, str, str]]:
    """Collect ``(source_citation_key, text, file_path)`` tuples for a claim comparison.

    For ``d_*`` keys, returns the single resolved text blob.
    For ``a_*`` keys (subagent outputs), also includes text from any upstream ``cited_in``
    sources, so claims can be checked against primary documents as well as the agent answer.
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

    return [(key, _citation_resolved_text(store, key), node.data.get("file_path")) for key in segment_keys]


# ---------------------------------------------------------------------------
# Internal helpers — text normalisation and chunking
# ---------------------------------------------------------------------------


def _normalize_for_overlap(text: str) -> str:
    """Lowercase and collapse whitespace for TF-IDF comparison."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _chunk_source_windows(
    source_norm: str,
    *,
    window_chars: int,
    stride_chars: int,
    max_chunks: int,
) -> list[str]:
    """Split *source_norm* into overlapping fixed-width windows for local similarity matching."""
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


def _split_sentences_simple(text: str) -> list[str]:
    """Split on ``.!?`` + whitespace; the last segment may lack terminal punctuation."""
    if not text or not text.strip():
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p for p in parts if p]


def _last_paragraph(text: str) -> str:
    """Return text after the last blank-line paragraph break, or the full string if none."""
    stripped = text.strip()
    if not stripped:
        return ""
    blocks = _PARAGRAPH_SPLIT_RE.split(stripped)
    return blocks[-1].strip() if blocks else stripped


# ---------------------------------------------------------------------------
# Public — Step 2 building blocks
# ---------------------------------------------------------------------------


def claim_source_tfidf_cosine(
    claim_text: str,
    source_text: str,
    *,
    max_source_chars: int = 96_000,
    chunk_chars: int = 1_200,
    chunk_stride: int = 600,
    max_chunks: int = 400,
) -> float:
    """Maximum TF-IDF cosine similarity between a claim and any window of a source.

    Splits the source into overlapping ``chunk_chars``-wide windows and builds a
    TF-IDF matrix over the claim plus all windows. Returns the highest per-window
    cosine similarity, clipped to ``[0, 1]``. Returns ``0.0`` for empty inputs or
    when there is no vocabulary overlap.
    """
    claim_text = _normalize_for_overlap(claim_text)
    source_text = _normalize_for_overlap(source_text[:max_source_chars])
    if not claim_text.strip() or not source_text.strip():
        return 0.0

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
    best = float(sims.max())
    if best != best:  # NaN when all TF-IDF vectors are zero
        return 0.0
    return min(1.0, max(0.0, best))


def context_before_span(
    text: str,
    start: int,
    *,
    max_chars: int = 720,
    max_sentences: int = 1,
) -> str:
    """Extract the claim context immediately before a citation span.

    Takes up to ``max_chars`` of text before ``start``, scopes to the last
    blank-line paragraph, applies simple ``.!?`` sentence splitting, and returns
    the last ``max_sentences`` sentences. Stops at any prior ``[REF|…]`` tag within
    the window so claims do not bleed across citation boundaries.

    Pass ``max_sentences=2`` for slightly more surrounding context.
    """
    prefix = text[:start]
    prefix_window = prefix[-max_chars:] if len(prefix) > max_chars else prefix

    # Find the last prior citation tag in the window; start context after it.
    prev_ref_match = None
    for match in re.finditer(r"\[REF\|[^\]]+\]", prefix_window):
        prev_ref_match = match

    body = prefix_window[prev_ref_match.end() :] if prev_ref_match else prefix_window
    body = body.strip()
    if not body:
        return ""

    scoped = _last_paragraph(body) or body
    sentences = _split_sentences_simple(scoped)
    if not sentences:
        return scoped

    tail = sentences[-max_sentences:] if len(sentences) > max_sentences else sentences
    return " ".join(s.strip() for s in tail).strip()


# ---------------------------------------------------------------------------
# Public — Step 3 entailment
# ---------------------------------------------------------------------------


class EntailmentJudge(Protocol):
    """Protocol for a custom Step 3 entailment judge.

    Implement this protocol to supply your own semantic support scorer.  The
    default implementation uses a pydantic-ai agent built by
    :func:`entailment_agent`.
    """

    async def score(self, *, source_excerpt: str, claim: str) -> tuple[float, str]:
        """Assess whether *source_excerpt* supports *claim*.

        Args:
            source_excerpt: A passage from the cited source document.
            claim: The author's claim or sentence immediately preceding the
                citation tag.

        Returns:
            A ``(probability, rationale)`` tuple where ``probability`` is a float
            in ``[0, 1]`` representing the degree of support and ``rationale`` is
            a brief human-readable explanation.
        """
        ...


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


def entailment_agent(model: Any, *, instructions: str | None = None) -> Any:
    """Build a pydantic-ai agent for Step 3 LLM-based entailment scoring.

    The caller supplies the model; the agent returns :class:`EntailmentJudgment`.
    """
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
    """Run the entailment agent for a single (source, claim) pair.

    Truncates both inputs to avoid exceeding the model's context window, then
    builds a structured prompt and awaits the agent's :class:`EntailmentJudgment`
    output.

    Args:
        agent: A pydantic-ai agent created by :func:`entailment_agent` (or any
            agent with a compatible ``run`` method returning ``EntailmentJudgment``).
        source_excerpt: The passage from the cited source used as the NLI premise.
        claim: The sentence or claim immediately preceding the citation tag, used
            as the NLI hypothesis.
        max_premise_chars: Maximum number of characters to pass as the premise.
        max_hypothesis_chars: Maximum number of characters to pass as the hypothesis.

    Returns:
        An :class:`EntailmentJudgment` with ``probability_support`` and
        ``rationale`` fields populated by the LLM.
    """
    premise = source_excerpt[:max_premise_chars]
    hypothesis = claim[:max_hypothesis_chars]
    prompt = f"Premise (source excerpt):\n{premise}\n\nHypothesis (cited claim):\n{hypothesis}"
    result = await agent.run(prompt)
    return result.output


# ---------------------------------------------------------------------------
# Public — Step 1: key sanitisation
# ---------------------------------------------------------------------------


def strip_unresolvable_citation_keys(text: str, store: ProvenanceStore) -> tuple[str, list[CitationKeyFilterResult]]:
    """Remove citation keys that don't resolve in the store; drop tags left empty.

    Processes tags in reverse order so character positions remain valid after
    each in-place replacement.
    """
    filter_records: list[CitationKeyFilterResult] = []
    sanitized_text = text

    for tag_start, tag_end, citation_tag in reversed(citation_tag_spans(text)):
        keys = citation_tag.refs
        kept = [k for k in keys if store.resolve_citation(k) is not None]

        if kept == keys:
            continue

        removed = list(set(keys) - set(kept))
        replacement_tag = "" if not kept else "[REF|" + "|".join(kept) + "]"
        sanitized_text = sanitized_text[:tag_start] + replacement_tag + sanitized_text[tag_end:]

        filter_records.append(
            CitationKeyFilterResult(
                raw_tag=citation_tag.raw,
                tag_start=tag_start,
                tag_end=tag_end,
                original_keys=list(keys),
                removed_keys=removed,
                retained_keys=list(kept),
            )
        )

    filter_records.reverse()
    return sanitized_text, filter_records


# ---------------------------------------------------------------------------
# Public — Step 2: TF-IDF overlap
# ---------------------------------------------------------------------------


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
    """Compute TF-IDF cosine similarity between each claim context and its cited sources.

    For every ``[REF|…]`` tag in *text*, extracts the sentence immediately before
    the tag as the claim, then scores it against each resolved source segment.
    """
    results: list[ClaimSourceSimilarity] = []

    for tag_start, tag_end, citation_tag in citation_tag_spans(text):
        claim_text = context_before_span(text, tag_start, max_chars=claim_max_chars)

        for claim_key in citation_tag.refs:
            source_segments = _gather_source_segments(store, claim_key)
            if not source_segments:
                continue

            source_keys, scores, source_excerpts, source_file_paths = [], [], [], []
            for source_key, source_text, source_file_path in source_segments:
                score = claim_source_tfidf_cosine(
                    claim_text=claim_text,
                    source_text=source_text,
                    max_source_chars=max_source_chars,
                    chunk_chars=chunk_chars,
                    chunk_stride=chunk_stride,
                    max_chunks=max_chunks,
                )
                source_keys.append(source_key)
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
    """For multi-key citation tags, keep only the top-N keys ranked by cosine score."""
    for record in records:
        if len(record.source_keys) <= top_n:
            continue

        ranked = sorted(
            zip(
                record.scores,
                record.source_keys,
                record.source_excerpts,
                record.source_file_paths,
                strict=False,
            ),
            key=lambda entry: (-entry[0], entry[1]),
        )
        top_entries = ranked[:top_n]

        record.scores = [e[0] for e in top_entries]
        record.source_keys = [e[1] for e in top_entries]
        record.source_excerpts = [e[2] for e in top_entries]
        record.source_file_paths = [e[3] for e in top_entries]
        record.verified_tag = "[REF|" + "|".join(record.source_keys) + "]"

    return records


def _drop_weak_sources(
    records: list[ClaimSourceSimilarity],
    min_score: float,
) -> list[ClaimSourceSimilarity]:
    """Drop citation keys whose similarity score falls below *min_score*."""
    for record in records:
        strong_keys, strong_scores, strong_excerpts, strong_paths = [], [], [], []
        for key, score, excerpt, path in zip(
            record.source_keys,
            record.scores,
            record.source_excerpts,
            record.source_file_paths,
            strict=False,
        ):
            if score >= min_score:
                strong_keys.append(key)
                strong_scores.append(score)
                strong_excerpts.append(excerpt)
                strong_paths.append(path)

        record.source_keys = strong_keys
        record.scores = strong_scores
        record.source_excerpts = strong_excerpts
        record.source_file_paths = strong_paths
        record.verified_tag = "[REF|" + "|".join(strong_keys) + "]" if strong_keys else ""

    return records


def refine_claim_source_similarities(
    records: list[ClaimSourceSimilarity],
    *,
    max_top_n_keys_per_tag: int = 2,
    min_score_for_shared_source: float = 0.3,
) -> list[ClaimSourceSimilarity]:
    """Narrow down similarity records for cleaner provenance signals.

    Two optional passes applied in order:

    1. **Min-score filter**: drop keys below ``min_score_for_shared_source``.
    2. **Top-N filter**: for multi-key tags, keep only the ``max_top_n_keys_per_tag``
       highest-scoring keys. Single-key tags are unchanged.
    """
    result = list(records)

    if min_score_for_shared_source is not None:
        result = _drop_weak_sources(result, min_score_for_shared_source)

    if max_top_n_keys_per_tag is not None and max_top_n_keys_per_tag > 0:
        result = _keep_top_citation_keys_per_tag(result, max_top_n_keys_per_tag)

    return result


def _remove_bad_citation_keys(
    text: str,
    similarities: list[ClaimSourceSimilarity],
) -> str:
    """Replace each citation tag in *text* with its verified form, or remove it if empty.

    Processes tags in reverse order by end position so earlier offsets stay valid.
    """
    for record in sorted(similarities, key=lambda r: r.tag_end, reverse=True):
        text = text[: record.tag_start] + record.verified_tag + text[record.tag_end :]
    return text


async def verify_citations(
    text: str,
    store: ProvenanceStore,
    *,
    # Step 2 — claim context extraction
    claim_context_chars: int = 720,
    # Step 2 — source TF-IDF chunking
    source_max_chars: int = 96_000,
    source_chunk_chars: int = 1_200,
    source_chunk_stride: int = 600,
    source_max_chunks: int = 400,
    # Step 2 — score filtering / refinement
    min_score: float = 0.3,
    max_keys_per_tag: int = 2,
) -> CitationVerificationReport:
    """Run Steps 1 and 2 of the citation verification pipeline.

    This is the primary entry point for citation verification.  It is ``async``
    so that Step 3 (LLM-based entailment scoring) can be added as an opt-in
    parameter in a future release without a breaking API change.  Steps 1 and 2
    themselves are CPU-bound and complete without any I/O.

    **Step 1 — key sanitisation**
    Every ``[REF|key]`` tag in *text* is checked against the store's citation
    registry.  Keys that cannot be resolved are removed from their tag; tags
    left with no valid keys are deleted entirely.

    **Step 2 — TF-IDF overlap scoring**
    For each remaining tag the sentence immediately before it is extracted as
    the *claim*.  That claim is compared via TF-IDF cosine similarity against
    the raw text of each cited source.  Tags whose best similarity score falls
    below *min_score* are removed; for multi-key tags the top *max_keys_per_tag*
    keys are retained and the rest are dropped.

    Args:
        text: The text containing inline ``[REF|…]`` citation tags to verify.
        store: The :class:`~pydantic_ai_provenance.store.ProvenanceStore` that
            maps citation keys to provenance nodes and their source text.
        claim_context_chars: Maximum number of characters extracted *before* each
            citation tag as the claim text for similarity scoring.  Smaller values
            narrow the claim to the immediately preceding sentence; larger values
            include more surrounding context.  Default: ``720``.
        source_max_chars: Maximum number of characters of source text fed into the
            TF-IDF vectoriser.  Documents longer than this are truncated before
            chunking.  Increase for very long documents at the cost of more memory.
            Default: ``96_000``.
        source_chunk_chars: Width of each overlapping window sliced from the
            (possibly truncated) source text for local similarity matching.  Should
            be comparable to the typical claim length.  Default: ``1_200``.
        source_chunk_stride: Step size between consecutive source windows.  A
            stride smaller than *source_chunk_chars* produces overlapping windows
            for better coverage at the cost of more chunks.  Default: ``600``.
        source_max_chunks: Maximum number of source windows created per citation
            key.  Caps memory and compute for very long documents.  Default: ``400``.
        min_score: TF-IDF cosine similarity threshold in ``[0, 1]``.  Citation
            keys whose best window score falls below this value are dropped from
            their tag.  Raise to be stricter (fewer, higher-confidence citations);
            lower to be more permissive.  Default: ``0.3``.
        max_keys_per_tag: Maximum number of citation keys retained per tag after
            scoring.  Keys are ranked by score (highest first) and only the top
            *max_keys_per_tag* are kept.  Single-key tags are not affected.
            Default: ``2``.

    Returns:
        A :class:`CitationVerificationReport` containing the original text, the
        text with verified (possibly narrowed) citation tags, and one
        :class:`ClaimSourceSimilarity` record per ``(tag, key)`` pair scored in
        Step 2.
    """
    sanitized_text, _ = strip_unresolvable_citation_keys(text, store)
    claim_source_similarities = compare_claim_to_sources_tfidf(
        sanitized_text,
        store,
        claim_max_chars=claim_context_chars,
        max_source_chars=source_max_chars,
        chunk_chars=source_chunk_chars,
        chunk_stride=source_chunk_stride,
        max_chunks=source_max_chunks,
    )
    refined_similarities = refine_claim_source_similarities(
        claim_source_similarities,
        min_score_for_shared_source=min_score,
        max_top_n_keys_per_tag=max_keys_per_tag,
    )
    text_with_verified_citations = _remove_bad_citation_keys(sanitized_text, refined_similarities)
    return CitationVerificationReport(
        original_text=text,
        text_with_verified_citations=text_with_verified_citations,
        claim_source_similarities=claim_source_similarities,
    )
