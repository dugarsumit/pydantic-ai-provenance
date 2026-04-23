"""Shared model-setup and display helpers used across the example scripts."""

from __future__ import annotations

import json
import os
import shutil
import textwrap
from textwrap import shorten
from typing import Any

from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.run import AgentRunResult

from pydantic_ai_provenance.store import ProvenanceStore
from pydantic_ai_provenance.verification import strip_unresolvable_citation_keys, verify_citations_sync

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def use_azure_openai() -> bool:
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"))


def _azure_chat_model() -> Any:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.azure import AzureProvider

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    return OpenAIChatModel(deployment, provider=AzureProvider())


def _anthropic_model(strong: bool) -> Any:
    from pydantic_ai.models.anthropic import AnthropicModel

    if strong:
        return AnthropicModel("claude-sonnet-4-6")
    return AnthropicModel("claude-haiku-4-5-20251001")


def example_model(*, strong: bool) -> Any:
    if use_azure_openai():
        return _azure_chat_model()
    return _anthropic_model(strong)


def require_credentials() -> None:
    """Exit with a helpful message when no LLM credentials are configured."""
    if not use_azure_openai() and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Configure credentials: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
            "for Azure OpenAI, or ANTHROPIC_API_KEY for Anthropic.",
            flush=True,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Terminal display helpers
# ---------------------------------------------------------------------------


def _term_width() -> int:
    try:
        return max(64, min(120, shutil.get_terminal_size().columns))
    except OSError:
        return 96


def _rule(char: str = "-", *, strong: bool = False) -> None:
    w = _term_width()
    print(("=" if strong else char) * w)


def _heading(lines: list[str], *, strong: bool = False) -> None:
    _rule(strong=strong)
    for line in lines:
        print(line)
    _rule(strong=strong)


def _maybe_pretty_json(s: str) -> str:
    t = s.strip()
    if not t:
        return s
    try:
        return json.dumps(json.loads(t), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        return s


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars - 80
    return text[:head] + f"\n\n… [{len(text) - head:,} more characters omitted; total {len(text):,}]"


def _print_body(text: str, *, max_chars: int = 16_000) -> None:
    body = _truncate(text, max_chars) if text else "(empty)"
    w = _term_width()
    prefix = "    "
    avail = max(20, w - len(prefix))
    for raw in body.splitlines():
        if len(raw) <= avail:
            print(f"{prefix}{raw}")
            continue
        wrapped = textwrap.fill(
            raw,
            width=avail,
            initial_indent=prefix,
            subsequent_indent=prefix,
            break_long_words=True,
            break_on_hyphens=False,
        )
        print(wrapped)
    print()


def _user_prompt_text(content: str | object) -> str:
    if isinstance(content, str):
        return content
    return repr(content)


# ---------------------------------------------------------------------------
# Model I/O printer
# ---------------------------------------------------------------------------


def print_model_io(result: AgentRunResult[Any], *, heading: str) -> None:
    """Print each model request/response from *result* (from ``result.new_messages()``)."""
    print()
    _heading([heading.center(_term_width())], strong=True)

    for i, msg in enumerate(result.new_messages(), 1):
        print()
        if isinstance(msg, ModelRequest):
            _heading(
                [
                    f"  [{i}]  MODEL REQUEST  ->  LLM",
                    f"      parts: {len(msg.parts)}",
                ]
            )
            if msg.instructions:
                print("  (instructions merged into this request)")
                _print_body(msg.instructions, max_chars=12_000)
            for j, part in enumerate(msg.parts, 1):
                sub = f"  [{i}.{j}]"
                _rule("-")
                if isinstance(part, SystemPromptPart):
                    print(f"{sub}  system prompt")
                    _print_body(part.content)
                elif isinstance(part, UserPromptPart):
                    print(f"{sub}  user prompt")
                    _print_body(_user_prompt_text(part.content))
                elif isinstance(part, ToolReturnPart | BuiltinToolReturnPart):
                    body = part.model_response_str() or repr(part.content)
                    print(f"{sub}  tool result  |  tool={part.tool_name!r}  tool_call_id={part.tool_call_id!r}")
                    _print_body(_maybe_pretty_json(body) if body.strip().startswith(("{", "[")) else body)
                elif isinstance(part, RetryPromptPart):
                    print(f"{sub}  retry prompt  |  tool={part.tool_name!r}")
                    _print_body(repr(part.content))
                else:
                    print(f"{sub}  {type(part).__name__}")
                    _print_body(repr(part))
        elif isinstance(msg, ModelResponse):
            u = msg.usage
            _heading(
                [
                    f"  [{i}]  MODEL RESPONSE  <-  LLM",
                    f"      model={msg.model_name!r}   tokens  in={u.request_tokens}  out={u.response_tokens}",
                ]
            )
            for j, part in enumerate(msg.parts, 1):
                sub = f"  [{i}.{j}]"
                _rule("-")
                if isinstance(part, TextPart):
                    print(f"{sub}  assistant text")
                    _print_body(part.content)
                elif isinstance(part, ToolCallPart | BuiltinToolCallPart):
                    raw_args = part.args_as_json_str()
                    print(f"{sub}  tool call  |  tool={part.tool_name!r}")
                    _print_body(_maybe_pretty_json(raw_args))
                elif isinstance(part, ThinkingPart) and part.content:
                    n = len(part.content)
                    cap = 4_000
                    body = part.content if n <= cap else part.content[:cap] + f"\n\n… [{n - cap:,} more chars]"
                    print(f"{sub}  thinking  ({n:,} chars)")
                    _print_body(body, max_chars=cap + 200)
                else:
                    print(f"{sub}  {type(part).__name__}")
                    _print_body(repr(part))
        else:
            _heading([f"  [{i}]  {type(msg).__name__}"])
            _print_body(repr(msg))


# ---------------------------------------------------------------------------
# Citation verification printer
# ---------------------------------------------------------------------------


def print_citation_verification(store: ProvenanceStore, *, label: str, text: str) -> None:
    """Print Step 1 (key sanitisation) and Step 2 (TF-IDF similarity) for *text*."""
    rep = verify_citations_sync(text, store)
    print("\n" + "-" * 60)
    print(f"Citation verification — {label}")
    print("-" * 60)
    _, sanitize_records = strip_unresolvable_citation_keys(text, store)
    if sanitize_records:
        print(f"Step 1: {len(sanitize_records)} tag(s) adjusted (``strip_unresolvable_citation_keys``)")
        for r in sanitize_records:
            print(f"  {r.raw_tag!r}: removed {r.removed_keys!r}, kept {r.retained_keys!r}")
    else:
        print("Step 1: all citation keys resolve in the store")
    if rep.text_with_verified_citations != rep.original_text:
        clip = (
            rep.text_with_verified_citations
            if len(rep.text_with_verified_citations) <= 400
            else rep.text_with_verified_citations[:400] + "…"
        )
        print(f"Text after weak-key removal (Step 2 refine):\n{clip}\n")
    rows = rep.claim_source_similarities
    print(f"Step 2: {len(rows)} claim↔source TF-IDF row(s)")
    headers = ["claim_key", "max_cos", "source_keys", "claim_ctx", "best_source_clip"]
    row_data: list[list[str]] = []
    for row in rows:
        scores = row.scores
        if scores:
            bi = max(range(len(scores)), key=lambda i: scores[i])
            max_cos = f"{scores[bi]:.3f}"
            best_src = shorten(row.source_excerpts[bi].replace("\n", " "), width=48, placeholder="…")
        else:
            max_cos = "—"
            best_src = "—"
        keys_s = ",".join(row.source_keys) if row.source_keys else "—"
        row_data.append(
            [
                row.claim_key,
                max_cos,
                shorten(keys_s, width=28, placeholder="…"),
                shorten(row.claim_excerpt.replace("\n", " "), width=40, placeholder="…"),
                best_src,
            ]
        )
    if not row_data:
        print("  (no resolvable inline [REF|…] spans with backing source text)")
        return
    col_widths = [max(len(str(r[i])) for r in ([headers] + row_data)) for i in range(len(headers))]

    def row_line(fields: list[str]) -> str:
        return " | ".join(str(f).ljust(col_widths[i]) for i, f in enumerate(fields))

    print(row_line(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in row_data:
        print(row_line(r))
