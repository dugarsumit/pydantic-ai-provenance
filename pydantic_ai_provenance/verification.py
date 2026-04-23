from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol, cast

from pydantic import BaseModel, Field

from .citations import citation_spans, is_agent_key, is_data_key
from .graph import NodeType, ProvenanceGraph
from .store import ProvenanceStore

try:
    from pydantic_ai import Agent as _PydanticAgent
except ImportError:  # pragma: no cover
    _PydanticAgent = None


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


def tool_result_raw_text(store: ProvenanceStore, source_call_node_id: str) -> str | None:
    """Return the raw tool result string for a DATA_READ call node."""
    graph = store.graph
    for succ in graph.successors(source_call_node_id):
        if succ.type == NodeType.TOOL_RESULT:
            raw = succ.data.get("result")
            return str(raw) if raw is not None else ""
    return None


def citation_resolved_text(store: ProvenanceStore, key: str) -> str | None:
    """Text blob associated with a citation key (for overlap / entailment checks)."""
    node_id = store.resolve_citation(key)
    if node_id is None:
        return None
    node = store.graph.nodes.get(node_id)
    if node is None:
        return None
    if node.type == NodeType.DATA_READ:
        raw = tool_result_raw_text(store, node_id)
        if raw is None:
            return None
        return _strip_ref_block_header(raw)
    if node.type == NodeType.FINAL_OUTPUT:
        out = node.data.get("output")
        return str(out) if out is not None else ""
    return None


def cited_in_source_node_ids(graph: ProvenanceGraph, target_node_id: str) -> list[str]:
    """Return source node ids that have a ``cited_in`` edge into ``target_node_id``."""
    return [
        e.source_id
        for e in graph.edges
        if e.target_id == target_node_id and e.label == "cited_in"
    ]


def expand_citation_texts_for_overlap(store: ProvenanceStore, key: str) -> list[tuple[str, str]]:
    """Return ``(key_or_leaf, text)`` segments to score for lexical overlap.

    For ``d_*`` (and legacy ``f_*`` / ``u_*``) returns a single segment. For ``a_*``
    (final output), includes that output plus any direct ``cited_in`` sources (e.g.
    underlying data keys) when those edges exist, so overlap can be checked against primary
    documents as well as the wrapped agent answer.
    """
    node_id = store.resolve_citation(key)
    if node_id is None:
        return []
    node = store.graph.nodes.get(node_id)
    if node is None:
        return []
    out: list[tuple[str, str]] = []

    def add_blob(k: str, nid: str) -> None:
        t = citation_resolved_text(store, k)
        if t is not None and t.strip():
            out.append((k, t))

    if node.type == NodeType.FINAL_OUTPUT:
        add_blob(key, node_id)
        for src_id in cited_in_source_node_ids(store.graph, node_id):
            leaf_key = store.citation_key_for_node(src_id)
            if not leaf_key:
                continue
            t = citation_resolved_text(store, leaf_key)
            if t is not None and t.strip():
                out.append((leaf_key, t))
    else:
        add_blob(key, node_id)

    # De-duplicate by key, keep order
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for k, t in out:
        if k not in seen:
            seen.add(k)
            deduped.append((k, t))
    return deduped


_WS_RE = re.compile(r"\s+")


def _normalize_for_overlap(s: str) -> str:
    return _WS_RE.sub(" ", s.strip().lower())


def best_lexical_support_ratio(claim: str, source: str, *, max_source_chars: int = 96_000) -> float:
    """Best ``SequenceMatcher`` ratio between normalized ``claim`` and a window of ``source``."""
    claim_n = _normalize_for_overlap(claim)
    if not claim_n:
        return 0.0
    src = source[:max_source_chars]
    src_n = _normalize_for_overlap(src)
    if not src_n:
        return 0.0
    nlen = len(claim_n)
    if len(src_n) <= nlen:
        return SequenceMatcher(None, claim_n, src_n).ratio()

    best = 0.0
    step = max(32, nlen // 3)
    max_i = len(src_n) - nlen + 1
    max_windows = 8_000
    for wi, i in enumerate(range(0, max_i, step)):
        if wi >= max_windows:
            break
        chunk = src_n[i : i + nlen]
        r = SequenceMatcher(None, claim_n, chunk).ratio()
        if r > best:
            best = r
    # slight size sweep around best-effort
    tail = src_n[-nlen:]
    best = max(best, SequenceMatcher(None, claim_n, tail).ratio())
    return float(best)


def context_before_span(text: str, start: int, *, max_chars: int = 720) -> str:
    """
    Get text before a citation tag, up to max_chars or stopping at the previous [REF|...] tag (within max_chars)
    or (NEW) at the previous line break.
    """
    prefix = text[:start]
    # Limit to max_chars window first
    if len(prefix) > max_chars:
        prefix_window = prefix[-max_chars:]
        window_start_in_prefix = len(prefix) - max_chars
    else:
        prefix_window = prefix
        window_start_in_prefix = 0

    import re

    # Find the last [REF|...]
    prev_ref_match = None
    for match in re.finditer(r"\[REF\|[^\]]+\]", prefix_window):
        prev_ref_match = match

    # Find the last line break in the prefix_window
    last_linebreak_idx = prefix_window.rfind('\n')

    # Figure out stopping points (previous [REF|...] tag or previous line break, whichever is nearer to end)
    stop_indices = []
    if prev_ref_match:
        stop_indices.append(prev_ref_match.end())
    if last_linebreak_idx != -1:
        stop_indices.append(last_linebreak_idx + 1)
    if stop_indices:
        # Use the furthest-forward stopping point (nearest to 'start')
        context_start_in_window = max(stop_indices)
        context_start = window_start_in_prefix + context_start_in_window
        return prefix[context_start:start].strip()
    else:
        return prefix_window.strip()


@dataclass
class KeySanitizeRecord:
    """One ``[REF|...]`` tag after Step 1 key validation."""

    raw_tag: str
    start: int
    end: int
    keys_in_tag: list[str]
    keys_removed: list[str]
    keys_kept: list[str]


@dataclass
class LexicalSupportRecord:
    """Step 2 fuzzy alignment for one (tag, key) pair."""

    citation_key: str
    source_key: str
    overlap_ratio: float
    claim_excerpt: str
    source_excerpt: str


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
    sanitized_text: str
    sanitize_records: list[KeySanitizeRecord] = field(default_factory=list)
    lexical: list[LexicalSupportRecord] = field(default_factory=list)
    entailment: list[EntailmentRecord] = field(default_factory=list)


class EntailmentJudge(Protocol):
    async def score(self, *, source_excerpt: str, claim: str) -> tuple[float, str]: ...


class EntailmentJudgment(BaseModel):
    """Structured output for the default entailment :class:`~pydantic_ai.agent.Agent`."""

    probability_support: float = Field(
        ge=0.0,
        le=1.0,
        description="How likely the premise supports the hypothesis (0–1).",
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


def sanitize_citation_keys(text: str, store: ProvenanceStore) -> tuple[str, list[KeySanitizeRecord]]:
    """Step 1: drop unknown keys from each tag; remove the tag if none remain."""
    records: list[KeySanitizeRecord] = []
    out = text
    for start, end, cref in reversed(citation_spans(text)):
        keys = cref.refs
        kept = [k for k in keys if store.resolve_citation(k) is not None]
        removed = [k for k in keys if k not in kept]
        if not kept:
            out = out[:start] + out[end:]
            records.append(
                KeySanitizeRecord(
                    raw_tag=cref.raw,
                    start=start,
                    end=end,
                    keys_in_tag=list(keys),
                    keys_removed=removed,
                    keys_kept=[],
                )
            )
        elif len(kept) == len(keys):
            continue
        else:
            new_tag = "[REF|" + "|".join(kept) + "]"
            out = out[:start] + new_tag + out[end:]
            records.append(
                KeySanitizeRecord(
                    raw_tag=cref.raw,
                    start=start,
                    end=end,
                    keys_in_tag=list(keys),
                    keys_removed=removed,
                    keys_kept=list(kept),
                )
            )
    records.reverse()
    return out, records


def run_lexical_verification(
    text: str,
    store: ProvenanceStore,
    *,
    context_max_chars: int = 720,
) -> list[LexicalSupportRecord]:
    """Step 2: for each tag and each kept key, best fuzzy overlap vs registered source text."""
    results: list[LexicalSupportRecord] = []
    for start, end, cref in citation_spans(text):
        claim = context_before_span(text, start, max_chars=context_max_chars)
        for key in cref.refs:
            if store.resolve_citation(key) is None:
                continue
            blobs = expand_citation_texts_for_overlap(store, key)
            if not blobs:
                continue
            best_ratio = -1.0
            best_sk = key
            best_excerpt = ""
            for sk, blob in blobs:
                r = best_lexical_support_ratio(claim, blob)
                if r > best_ratio:
                    best_ratio = r
                    best_sk = sk
                    best_excerpt = blob.replace("\n", " ")
            results.append(
                LexicalSupportRecord(
                    citation_key=key,
                    source_key=best_sk,
                    overlap_ratio=float(best_ratio) if best_ratio >= 0 else 0.0,
                    claim_excerpt=claim,
                    source_excerpt=best_excerpt,
                )
            )
    return results


async def verify_citations(
    text: str,
    store: ProvenanceStore,
    *,
    context_max_chars: int = 720,
    entailment_judge: Any | None = None,
    max_entailment_source_chars: int = 6_000,
) -> CitationVerificationReport:
    """Run Step 1 (always), Step 2 (always), and Step 3 when ``entailment_judge`` is set.

    ``entailment_judge`` may be an :func:`entailment_agent` instance or any object
    with ``async def score(*, source_excerpt, claim) -> tuple[float, str]``.
    """
    sanitized, sanitize_records = sanitize_citation_keys(text, store)
    lexical = run_lexical_verification(sanitized, store, context_max_chars=context_max_chars)
    entailment: list[EntailmentRecord] = []

    if entailment_judge is not None:
        judge = entailment_judge
        use_pydantic_agent = _PydanticAgent is not None and isinstance(judge, _PydanticAgent)
        for rec in lexical:
            blob = citation_resolved_text(store, rec.citation_key)
            if blob is None:
                for sk, _ in expand_citation_texts_for_overlap(store, rec.citation_key):
                    blob = citation_resolved_text(store, sk)
                    if blob:
                        break
            if not blob:
                continue
            excerpt = blob[:max_entailment_source_chars]
            if use_pydantic_agent:
                judgment = await _score_with_pydantic_agent(
                    judge,
                    source_excerpt=excerpt,
                    claim=rec.claim_excerpt,
                )
                entailment.append(
                    EntailmentRecord(
                        citation_key=rec.citation_key,
                        probability=judgment.probability_support,
                        rationale=judgment.rationale,
                    )
                )
            else:
                protocol_judge = cast(EntailmentJudge, judge)
                prob, rationale = await protocol_judge.score(
                    source_excerpt=excerpt,
                    claim=rec.claim_excerpt,
                )
                prob = max(0.0, min(1.0, float(prob)))
                entailment.append(
                    EntailmentRecord(
                        citation_key=rec.citation_key,
                        probability=prob,
                        rationale=rationale,
                    )
                )

    return CitationVerificationReport(
        original_text=text,
        sanitized_text=sanitized,
        sanitize_records=sanitize_records,
        lexical=lexical,
        entailment=entailment,
    )


def verify_citations_sync(
    text: str,
    store: ProvenanceStore,
    *,
    context_max_chars: int = 720,
) -> CitationVerificationReport:
    """Steps 1-2 only (no LLM). Async :func:`verify_citations` adds Step 3."""
    sanitized, sanitize_records = sanitize_citation_keys(text, store)
    lexical = run_lexical_verification(sanitized, store, context_max_chars=context_max_chars)
    return CitationVerificationReport(
        original_text=text,
        sanitized_text=sanitized,
        sanitize_records=sanitize_records,
        lexical=lexical,
        entailment=[],
    )


# Re-export key-type helpers for callers building UIs
def citation_key_kind(key: str) -> str:
    if is_data_key(key):
        return "data"
    if is_agent_key(key):
        return "agent"
    return "unknown"
