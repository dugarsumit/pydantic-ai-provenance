from .attribution import AttributionPath, AttributionResult, attribute_all_outputs, attribute_output
from .capability import ProvenanceCapability
from .citations import (
    CitationRef,
    citation_tag_spans,
    parse_citations,
    strip_inline_citation_tags,
    strip_inline_citation_tags_preserve_leading_ref_header,
)
from .graph import NodeType, ProvenanceEdge, ProvenanceGraph, ProvenanceNode
from .store import ProvenanceStore
from .verification import (
    CitationKeyFilterResult,
    CitationVerificationReport,
    ClaimSourceSimilarity,
    EntailmentJudgment,
    EntailmentJudge,
    EntailmentRecord,
    _citation_resolved_text,
    claim_source_tfidf_cosine,
    compare_claim_to_sources_tfidf,
    context_before_span,
    entailment_agent,
    refine_claim_source_similarities,
    strip_unresolvable_citation_keys,
    _tool_result_raw_text,
    verify_citations_sync,
)
from .viz import to_dot, to_json, to_json_str, to_mermaid

__all__ = [
    # Core
    "ProvenanceCapability",
    "ProvenanceStore",
    # Graph primitives
    "ProvenanceGraph",
    "ProvenanceNode",
    "ProvenanceEdge",
    "NodeType",
    # Attribution
    "attribute_output",
    "attribute_all_outputs",
    "AttributionResult",
    "AttributionPath",
    # Citations
    "CitationRef",
    "citation_tag_spans",
    "parse_citations",
    "strip_inline_citation_tags",
    "strip_inline_citation_tags_preserve_leading_ref_header",
    # Citation verification
    "CitationVerificationReport",
    "EntailmentJudgment",
    "EntailmentJudge",
    "EntailmentRecord",
    "CitationKeyFilterResult",
    "ClaimSourceSimilarity",
    "_citation_resolved_text",
    "claim_source_tfidf_cosine",
    "compare_claim_to_sources_tfidf",
    "context_before_span",
    "entailment_agent",
    "refine_claim_source_similarities",
    "strip_unresolvable_citation_keys",
    "_tool_result_raw_text",
    "verify_citations_sync",
    # Visualization
    "to_mermaid",
    "to_dot",
    "to_json",
    "to_json_str",
]
