# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-01

### Added
- `ProvenanceCapability`: pydantic-ai `AbstractCapability` that tracks full agent execution as a DAG.
- `ProvenanceStore`: central citation registry shared across single-agent and multi-agent sessions.
- `ProvenanceGraph` / `ProvenanceNode` / `ProvenanceEdge`: typed graph primitives.
- `NodeType` enum: `INPUT`, `DATA_READ`, `TOOL_CALL`, `TOOL_RESULT`, `MODEL_REQUEST`, `MODEL_RESPONSE`, `AGENT_RUN`, `FINAL_OUTPUT`.
- Citation format `[REF|key1|key2|...]` for inline source attribution.
- `parse_citations`, `strip_inline_citation_tags`, `citation_tag_spans`: citation string helpers.
- `verify_citations_sync`: two-step verification — key sanitisation (Step 1) + TF-IDF cosine overlap (Step 2).
- `claim_source_tfidf_cosine`: sliding-window TF-IDF similarity for long sources.
- `entailment_agent`: optional LLM-based entailment judge (Step 3) via pydantic-ai.
- `attribute_output` / `attribute_all_outputs`: path-level attribution from source nodes to final outputs.
- `to_mermaid`, `to_dot`, `to_json`, `to_json_str`: provenance graph visualisation.
- Multi-agent support: subagent outputs propagated through shared store via `contextvars`.

[Unreleased]: https://github.com/dugarsumit/pydantic-ai-provenance/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dugarsumit/pydantic-ai-provenance/releases/tag/v0.1.0
