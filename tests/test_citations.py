"""Tests for pydantic_ai_provenance.citations."""

from __future__ import annotations

from pydantic_ai_provenance.citations import (
    CitationRef,
    citation_tag_spans,
    extract_file_path,
    format_cited_content,
    is_agent_key,
    is_data_key,
    parse_citations,
    strip_inline_citation_tags,
    strip_inline_citation_tags_preserve_leading_ref_header,
)

# ---------------------------------------------------------------------------
# CitationRef
# ---------------------------------------------------------------------------


def test_citation_ref_str_single():
    ref = CitationRef(refs=["d_1"], raw="[REF|d_1]")
    assert str(ref) == "[REF|d_1]"


def test_citation_ref_str_multi():
    ref = CitationRef(refs=["d_1", "a_2"], raw="[REF|d_1|a_2]")
    assert str(ref) == "[REF|d_1|a_2]"


# ---------------------------------------------------------------------------
# Key prefix helpers
# ---------------------------------------------------------------------------


def test_is_data_key_true():
    assert is_data_key("d_1") is True


def test_is_data_key_false():
    assert is_data_key("a_1") is False


def test_is_agent_key_true():
    assert is_agent_key("a_3") is True


def test_is_agent_key_false():
    assert is_agent_key("d_3") is False


# ---------------------------------------------------------------------------
# parse_citations
# ---------------------------------------------------------------------------


def test_parse_citations_empty_string():
    assert parse_citations("") == []


def test_parse_citations_no_tags():
    assert parse_citations("No citation here.") == []


def test_parse_citations_single_key():
    result = parse_citations("Some claim. [REF|d_1]")
    assert len(result) == 1
    assert result[0].refs == ["d_1"]
    assert result[0].raw == "[REF|d_1]"


def test_parse_citations_multiple_keys_one_tag():
    result = parse_citations("Claim. [REF|d_1|a_2]")
    assert len(result) == 1
    assert result[0].refs == ["d_1", "a_2"]


def test_parse_citations_multiple_tags():
    text = "First fact. [REF|d_1] Second fact. [REF|a_2]"
    result = parse_citations(text)
    assert len(result) == 2
    assert result[0].refs == ["d_1"]
    assert result[1].refs == ["a_2"]


def test_parse_citations_invalid_format_not_matched():
    assert parse_citations("[ref|d_1]") == []
    assert parse_citations("[REF d_1]") == []
    assert parse_citations("[REF|]") == []


# ---------------------------------------------------------------------------
# strip_inline_citation_tags
# ---------------------------------------------------------------------------


def test_strip_inline_citation_tags_removes_all():
    text = "Claim. [REF|d_1] Another claim. [REF|a_2|d_3]"
    result = strip_inline_citation_tags(text)
    assert "[REF|" not in result
    assert "Claim." in result
    assert "Another claim." in result


def test_strip_inline_citation_tags_empty_string():
    assert strip_inline_citation_tags("") == ""


def test_strip_inline_citation_tags_no_tags():
    text = "No citations here."
    assert strip_inline_citation_tags(text) == text


# ---------------------------------------------------------------------------
# strip_inline_citation_tags_preserve_leading_ref_header
# ---------------------------------------------------------------------------


def test_preserve_header_keeps_first_line_tag():
    text = "[REF|d_1]\nSome body with [REF|d_1] inline."
    result = strip_inline_citation_tags_preserve_leading_ref_header(text)
    assert result.startswith("[REF|d_1]")
    assert "[REF|d_1]\nSome body with " in result
    assert result.endswith("inline.")


def test_preserve_header_strips_body_tags():
    text = "[REF|d_1]\nBody text [REF|d_2] with inline tag."
    result = strip_inline_citation_tags_preserve_leading_ref_header(text)
    assert result.startswith("[REF|d_1]")
    assert "[REF|d_2]" not in result


def test_preserve_header_no_header_strips_all():
    text = "No header. [REF|d_1] inline."
    result = strip_inline_citation_tags_preserve_leading_ref_header(text)
    assert "[REF|d_1]" not in result


def test_preserve_header_only_one_line_tag():
    text = "[REF|d_1]"
    result = strip_inline_citation_tags_preserve_leading_ref_header(text)
    assert result == "[REF|d_1]"


def test_preserve_header_first_line_not_a_clean_tag():
    text = "[REF|d_1] extra words on header line\nBody [REF|d_2]."
    result = strip_inline_citation_tags_preserve_leading_ref_header(text)
    assert "[REF|d_1]" not in result
    assert "[REF|d_2]" not in result


# ---------------------------------------------------------------------------
# citation_tag_spans
# ---------------------------------------------------------------------------


def test_citation_tag_spans_returns_positions():
    text = "Claim A. [REF|d_1] Claim B. [REF|a_2]"
    spans = citation_tag_spans(text)
    assert len(spans) == 2
    start1, end1, ref1 = spans[0]
    assert text[start1:end1] == "[REF|d_1]"
    assert ref1.refs == ["d_1"]


def test_citation_tag_spans_empty():
    assert citation_tag_spans("No tags here.") == []


def test_citation_tag_spans_multi_key():
    text = "[REF|d_1|a_2]"
    spans = citation_tag_spans(text)
    assert len(spans) == 1
    start, end, ref = spans[0]
    assert start == 0
    assert end == len(text)
    assert ref.refs == ["d_1", "a_2"]


# ---------------------------------------------------------------------------
# extract_file_path
# ---------------------------------------------------------------------------


def test_extract_file_path_priority_path():
    assert extract_file_path({"path": "/a.txt", "file": "/b.txt"}) == "/a.txt"


def test_extract_file_path_priority_file():
    assert extract_file_path({"file": "/b.txt", "url": "http://x"}) == "/b.txt"


def test_extract_file_path_priority_filename():
    assert extract_file_path({"filename": "report.csv"}) == "report.csv"


def test_extract_file_path_priority_url():
    assert extract_file_path({"url": "http://example.com"}) == "http://example.com"


def test_extract_file_path_fallback_first_string():
    assert extract_file_path({"query": "something"}) == "something"


def test_extract_file_path_empty_dict():
    assert extract_file_path({}) is None


def test_extract_file_path_no_string_values():
    assert extract_file_path({"count": 42, "flag": True}) is None


# ---------------------------------------------------------------------------
# format_cited_content
# ---------------------------------------------------------------------------


def test_format_cited_content_simple():
    result = format_cited_content("hello world", "d_1")
    assert result == "[REF|d_1]\nhello world"


def test_format_cited_content_converts_to_str():
    result = format_cited_content(42, "a_3")
    assert result == "[REF|a_3]\n42"


def test_format_cited_content_preserves_newlines_in_body():
    result = format_cited_content("line1\nline2", "d_2")
    assert result == "[REF|d_2]\nline1\nline2"
