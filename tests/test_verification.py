"""Tests for pydantic_ai_provenance.verification."""

from __future__ import annotations

import pytest

from pydantic_ai_provenance.graph import NodeType, ProvenanceNode
from pydantic_ai_provenance.store import ProvenanceStore
from pydantic_ai_provenance.verification import (
    CitationVerificationReport,
    ClaimSourceSimilarity,
    claim_source_tfidf_cosine,
    context_before_span,
    refine_claim_source_similarities,
    strip_unresolvable_citation_keys,
    verify_citations,
)


def _make_node(node_type: NodeType, label: str, **data) -> ProvenanceNode:
    return ProvenanceNode.create(type=node_type, label=label, agent_name="agent", run_id="r", **data)


def _build_store_with_source(source_text: str) -> tuple[ProvenanceStore, str]:
    """DATA_READ → TOOL_RESULT. Returns (store, citation_key)."""
    store = ProvenanceStore()
    data_node = _make_node(NodeType.DATA_READ, "read_file", file_path="test.txt")
    store.add_node(data_node)
    citation_key = store.register_data_source(data_node.id)

    result_node = _make_node(
        NodeType.TOOL_RESULT,
        "Result: read_file",
        tool_name="read_file",
        result=source_text,
    )
    store.add_node(result_node)
    store.add_edge(data_node.id, result_node.id, "returns")
    return store, citation_key


# ---------------------------------------------------------------------------
# claim_source_tfidf_cosine
# ---------------------------------------------------------------------------


def test_tfidf_cosine_identical_texts():
    text = "the quick brown fox jumps over the lazy dog"
    score = claim_source_tfidf_cosine(text, text)
    assert score > 0.9


def test_tfidf_cosine_empty_claim_returns_zero():
    assert claim_source_tfidf_cosine("", "some source text") == 0.0


def test_tfidf_cosine_empty_source_returns_zero():
    assert claim_source_tfidf_cosine("some claim", "") == 0.0


def test_tfidf_cosine_both_empty_returns_zero():
    assert claim_source_tfidf_cosine("", "") == 0.0


def test_tfidf_cosine_unrelated_texts_low_score():
    score = claim_source_tfidf_cosine(
        "quantum entanglement and superluminal communication",
        "the quick brown fox jumps over the lazy dog near the river",
    )
    assert score < 0.3


def test_tfidf_cosine_result_in_range():
    score = claim_source_tfidf_cosine(
        "pydantic ai framework", "pydantic ai is a powerful framework for building agents"
    )
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# context_before_span
# ---------------------------------------------------------------------------


def test_context_before_span_basic():
    text = "The fox is quick. [REF|d_1]"
    tag_pos = text.index("[REF|d_1]")
    ctx = context_before_span(text, tag_pos)
    assert "fox" in ctx


def test_context_before_span_empty_before_tag():
    text = "[REF|d_1] something after"
    ctx = context_before_span(text, 0)
    assert ctx == ""


def test_context_before_span_stops_at_prior_ref():
    text = "First sentence. [REF|d_1] Irrelevant. Second claim. [REF|d_2]"
    tag_pos = text.index("[REF|d_2]")
    ctx = context_before_span(text, tag_pos)
    # Should be extracted from after the first [REF|d_1] tag
    assert "First sentence" not in ctx
    assert "Second claim" in ctx


def test_context_before_span_max_chars_truncation():
    long_prefix = "word " * 1000  # 5000 chars
    text = long_prefix + "[REF|d_1]"
    tag_pos = len(long_prefix)
    ctx = context_before_span(text, tag_pos, max_chars=100)
    assert len(ctx) <= 100


# ---------------------------------------------------------------------------
# strip_unresolvable_citation_keys
# ---------------------------------------------------------------------------


def test_strip_unresolvable_all_valid():
    store, key = _build_store_with_source("fox")
    text = f"A claim. [REF|{key}]"
    sanitized, records = strip_unresolvable_citation_keys(text, store)
    assert sanitized == text
    assert records == []


def test_strip_unresolvable_unknown_key_removed():
    store, _ = _build_store_with_source("fox")
    text = "Claim. [REF|bad_key]"
    sanitized, records = strip_unresolvable_citation_keys(text, store)
    assert "[REF|bad_key]" not in sanitized
    assert len(records) == 1
    assert "bad_key" in records[0].removed_keys


def test_strip_unresolvable_mixed_keeps_valid():
    store, key = _build_store_with_source("fox")
    text = f"Claim. [REF|{key}|fake_key]"
    sanitized, records = strip_unresolvable_citation_keys(text, store)
    assert f"[REF|{key}]" in sanitized
    assert "fake_key" not in sanitized
    assert len(records) == 1


def test_strip_unresolvable_all_invalid_tag_removed():
    store, _ = _build_store_with_source("fox")
    text = "Claim. [REF|bad1|bad2]"
    sanitized, records = strip_unresolvable_citation_keys(text, store)
    assert "[REF|" not in sanitized
    assert len(records) == 1
    assert records[0].retained_keys == []


# ---------------------------------------------------------------------------
# verify_citations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_verify_citations_returns_report():
    store, key = _build_store_with_source("The quick brown fox jumps over the lazy dog.")
    text = f"A fox appears. [REF|{key}]"
    report = await verify_citations(text, store)
    assert isinstance(report, CitationVerificationReport)
    assert report.original_text == text


@pytest.mark.asyncio
async def test_async_verify_citations_aligned_claim_keeps_tag():
    store, key = _build_store_with_source("The quick brown fox jumps over the lazy dog near the river.")
    text = f"The passage mentions a quick brown fox near the river. [REF|{key}]"
    report = await verify_citations(text, store)
    assert f"[REF|{key}]" in report.text_with_verified_citations


@pytest.mark.asyncio
async def test_async_verify_citations_bogus_key_removed():
    store, _ = _build_store_with_source("fox")
    text = "Claim. [REF|totally_fake]"
    report = await verify_citations(text, store)
    assert "[REF|totally_fake]" not in report.text_with_verified_citations


@pytest.mark.asyncio
async def test_async_verify_citations_no_tags_passthrough():
    store, _ = _build_store_with_source("fox")
    text = "Plain text without any citation tags."
    report = await verify_citations(text, store)
    assert report.text_with_verified_citations == text
    assert report.claim_source_similarities == []


# ---------------------------------------------------------------------------
# refine_claim_source_similarities
# ---------------------------------------------------------------------------


def _make_similarity(claim_key: str, source_keys: list[str], scores: list[float]) -> ClaimSourceSimilarity:
    return ClaimSourceSimilarity(
        method="tfidf",
        claim_key=claim_key,
        source_keys=source_keys,
        scores=scores,
        claim_excerpt="some claim",
        source_excerpts=["excerpt"] * len(source_keys),
        source_file_paths=[""] * len(source_keys),
        raw_tag="[REF|" + "|".join(source_keys) + "]",
        verified_tag="[REF|" + "|".join(source_keys) + "]",
    )


def test_refine_drops_weak_sources_below_min_score():
    rec = _make_similarity("d_1", ["d_1", "d_2"], [0.8, 0.1])
    result = refine_claim_source_similarities([rec], min_score_for_shared_source=0.3)
    assert result[0].source_keys == ["d_1"]


def test_refine_keeps_strong_sources():
    rec = _make_similarity("d_1", ["d_1", "d_2"], [0.8, 0.6])
    result = refine_claim_source_similarities([rec], min_score_for_shared_source=0.3, max_top_n_keys_per_tag=5)
    assert "d_1" in result[0].source_keys
    assert "d_2" in result[0].source_keys


def test_refine_top_n_limits_keys():
    rec = _make_similarity("d_1", ["d_1", "d_2", "d_3"], [0.9, 0.8, 0.7])
    result = refine_claim_source_similarities([rec], max_top_n_keys_per_tag=2, min_score_for_shared_source=0.0)
    assert len(result[0].source_keys) <= 2
