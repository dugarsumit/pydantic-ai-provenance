"""Citation tag parsing, formatting, and key-classification helpers.

The citation format used throughout the library is ``[REF|key1|key2|…]``.
Keys are short identifiers assigned by :class:`~.store.ProvenanceStore` at
source-registration time (e.g. ``d_1`` for data sources, ``a_1`` for agent outputs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches [REF|d_1], [REF|a_1], [REF|d_1|d_2], [REF|d_1|a_2], etc.
# Single brackets, REF sentinel, then one or more pipe-separated citation keys.
# Keys are valid identifiers (letter/underscore start, alphanumeric/underscore body).
_CITATION_RE = re.compile(r"\[REF\|([a-zA-Z_][a-zA-Z0-9_]*(?:\|[a-zA-Z_][a-zA-Z0-9_]*)*)\]")

# Arg names tried in order when auto-detecting the file path from tool args.
_FILE_ARG_PRIORITY = ("path", "file", "filename", "filepath", "url", "uri", "source")

# Citation key prefixes — baked into store-generated keys so the graph is
# human-readable without needing to look up the registry.
KEY_PREFIX_DATA = "d"
KEY_PREFIX_AGENT = "a"


def is_data_key(key: str) -> bool:
    """Return ``True`` if *key* is a data-source key (``d_*``)."""
    return key.startswith(KEY_PREFIX_DATA + "_")


def is_agent_key(key: str) -> bool:
    """Return ``True`` if *key* is an agent-output key (``a_*``)."""
    return key.startswith(KEY_PREFIX_AGENT + "_")


@dataclass
class CitationRef:
    """A single ``[REF|…]`` tag parsed from an LLM response.

    One tag can cite multiple sources at once::

        [REF|d_1|d_2|a_3]

    Each key in ``refs`` is an opaque identifier (e.g. ``"d_1"``, ``"a_2"``)
    assigned by the store at source-registration time.  Resolve any key via
    :meth:`~.store.ProvenanceStore.resolve_citation` to get its
    :class:`~.graph.ProvenanceNode` and full metadata.

    No line numbers or other metadata live in the tag itself — everything is in
    the graph.
    """

    refs: list[str]
    raw: str

    def __str__(self) -> str:
        return f"[REF|{'|'.join(self.refs)}]"


def parse_citations(text: str) -> list[CitationRef]:
    """Extract all ``[REF|key1|key2|…]`` citation tags from *text*."""
    return [CitationRef(refs=match.group(1).split("|"), raw=match.group(0)) for match in _CITATION_RE.finditer(text)]


def strip_inline_citation_tags(text: str) -> str:
    """Remove every ``[REF|…]`` tag from *text*. Returns *text* unchanged if falsy."""
    if not text:
        return text
    return _CITATION_RE.sub("", text)


def strip_inline_citation_tags_preserve_leading_ref_header(text: str) -> str:
    """Remove inline ``[REF|…]`` tags but keep one optional opening block-header line.

    If the first line of *text* is exactly a single ``[REF|key|…]`` tag (the
    wrapped tool/subagent block header written by :func:`format_cited_content`),
    that line is kept and only the following body is stripped.

    Otherwise identical to :func:`strip_inline_citation_tags`.
    """
    if not text:
        return text
    if text.startswith("[REF|"):
        nl = text.find("\n")
        if nl == -1:
            first, rest = text.strip(), ""
        else:
            first, rest = text[:nl].strip(), text[nl + 1 :]
        if _CITATION_RE.fullmatch(first):
            return first + ("\n" if rest else "") + strip_inline_citation_tags(rest)
    return strip_inline_citation_tags(text)


def citation_tag_spans(text: str) -> list[tuple[int, int, CitationRef]]:
    """Like :func:`parse_citations` but also returns each tag's ``(start, end)`` positions."""
    return [
        (
            match.start(),
            match.end(),
            CitationRef(refs=match.group(1).split("|"), raw=match.group(0)),
        )
        for match in _CITATION_RE.finditer(text)
    ]


def extract_file_path(args: dict[str, object]) -> str | None:
    """Find a file-path argument from a source tool's validated args dict.

    Tries the keys in :data:`_FILE_ARG_PRIORITY` order, then falls back to the
    first string-valued argument. Returns ``None`` if no string value is found.
    """
    for name in _FILE_ARG_PRIORITY:
        if name in args:
            return str(args[name])
    for value in args.values():
        if isinstance(value, str):
            return value
    return None


def format_cited_content(result: object, citation_key: str) -> str:
    """Wrap tool or subagent output in a ``[REF|<key>]`` block header.

    The returned string always starts with ``[REF|<citation_key>]`` on its own
    line, followed by ``str(result)``.  The model is expected to use this key
    when citing content from this result in its final response.
    """
    return f"[REF|{citation_key}]\n{result}"
