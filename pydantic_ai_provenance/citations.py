from __future__ import annotations

import re
from dataclasses import dataclass

# Matches [REF|f_1], [REF|u_1], [REF|a_1], [REF|f_1|f_2], [REF|f_1|a_2], etc.
# Single brackets, REF sentinel, then one or more pipe-separated citation keys.
# Keys are valid identifiers (letter/underscore start, alphanumeric/underscore body).
_CITATION_RE = re.compile(
    r'\[REF\|([a-zA-Z_][a-zA-Z0-9_]*(?:\|[a-zA-Z_][a-zA-Z0-9_]*)*)\]'
)

# Arg names tried in order when auto-detecting the file path from tool args.
_FILE_ARG_PRIORITY = ("path", "file", "filename", "filepath", "url", "uri", "source")

# Citation key prefixes — baked into store-generated keys so the graph is
# human-readable without needing to look up the registry.
KEY_PREFIX_FILE = "f"
KEY_PREFIX_AGENT = "a"
KEY_PREFIX_URL = "u"


def is_file_key(key: str) -> bool:
    return key.startswith(KEY_PREFIX_FILE + "_")

def is_agent_key(key: str) -> bool:
    return key.startswith(KEY_PREFIX_AGENT + "_")

def is_url_key(key: str) -> bool:
    return key.startswith(KEY_PREFIX_URL + "_")


@dataclass
class CitationRef:
    """A single [REF|...] tag parsed from an LLM response.

    One tag can cite multiple sources at once:
        [REF|f_1|f_2|a_3]

    Each key in `refs` is an opaque identifier (e.g. "f_1", "a_2")
    assigned by the store at source-registration time. Resolve any key via
    ProvenanceStore.resolve_citation() to get its ProvenanceNode and full
    metadata (file path, agent name, run_id, etc.).

    No line numbers or other metadata live in the tag — everything is in the graph.
    """

    refs: list[str]
    raw: str

    def __str__(self) -> str:
        return f"[REF|{'|'.join(self.refs)}]"


def parse_citations(text: str) -> list[CitationRef]:
    """Extract all [REF|key1|key2|...] citation tags from a text string."""
    return [
        CitationRef(refs=match.group(1).split("|"), raw=match.group(0))
        for match in _CITATION_RE.finditer(text)
    ]


def strip_inline_citation_tags(text: str) -> str:
    """Remove every ``[REF|...]`` tag from *text* (empty string if *text* is falsy)."""
    if not text:
        return text
    return _CITATION_RE.sub("", text)


def strip_inline_citation_tags_preserve_leading_ref_header(text: str) -> str:
    """Remove inline ``[REF|...]`` tags but keep one optional opening source header line.

    If the first line is exactly a single ``[REF|key|...]`` tag, that line is kept
    (the wrapped tool/subagent block header) and tags only in the following body
    are stripped.

    Otherwise identical to :func:`strip_inline_citation_tags`.
    """
    if not text:
        return text
    if text.startswith("[REF|"):
        nl = text.find("\n")
        if nl == -1:
            first = text.strip()
            rest = ""
        else:
            first = text[:nl].strip()
            rest = text[nl + 1 :]
        if _CITATION_RE.fullmatch(first):
            return first + ("\n" if rest else "") + strip_inline_citation_tags(rest)
        return strip_inline_citation_tags(text)
    return strip_inline_citation_tags(text)


def citation_spans(text: str) -> list[tuple[int, int, CitationRef]]:
    """Like parse_citations but preserves each match's start/end indices in ``text``."""
    return [
        (
            match.start(),
            match.end(),
            CitationRef(refs=match.group(1).split("|"), raw=match.group(0)),
        )
        for match in _CITATION_RE.finditer(text)
    ]


def extract_file_path(args: dict[str, object]) -> str | None:
    """Find the file-path argument from a source tool's validated args."""
    for name in _FILE_ARG_PRIORITY:
        if name in args:
            return str(args[name])
    for v in args.values():
        if isinstance(v, str):
            return v
    return None


def format_cited_content(result: object, citation_key: str) -> str:
    """Wrap tool or subagent output in a ``[REF|<key>]`` block.

    The first line is always ``[REF|<citation_key>]``; any **inline**
    ``[REF|...]`` tags inside *result* are stripped before wrapping so nested
    payloads do not duplicate markers in the body.
    """
    content = strip_inline_citation_tags(str(result))
    return f"[REF|{citation_key}]\n{content}"
