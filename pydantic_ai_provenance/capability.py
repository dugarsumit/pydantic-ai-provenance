from __future__ import annotations

import uuid
from contextvars import Token
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition

from .citations import (
    extract_file_path,
    format_cited_content,
    parse_citations,
    strip_inline_citation_tags,
    strip_inline_citation_tags_preserve_leading_ref_header,
)
from .graph import NodeType, ProvenanceNode
from .store import ProvenanceStore, _PROVENANCE_CTX

def _strip_inline_refs_from_request_context(request_context: Any) -> None:
    """Remove inline ``[REF|...]`` tags from message parts except static instructions.

    Keeps system and instruction parts intact. Strips all tags from user text,
    assistant text, thinking, etc. For **tool return** parts, keeps a single opening
    ``[REF|<key>]`` header line and
    strips only tags in the body below it.
    """
    from pydantic_ai.messages import (
        InstructionPart,
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        ToolCallPart,
        ToolReturnPart,
    )

    for msg in request_context.messages:
        if isinstance(msg, ModelRequest):
            parts = msg.parts
        elif isinstance(msg, ModelResponse):
            parts = msg.parts
        else:
            continue
        for part in parts:
            if isinstance(part, (SystemPromptPart, InstructionPart)):
                continue
            if isinstance(part, ToolCallPart):
                if isinstance(part.args, str):
                    part.args = strip_inline_citation_tags(part.args)
                continue
            if isinstance(part, ToolReturnPart):
                if isinstance(part.content, str):
                    part.content = strip_inline_citation_tags_preserve_leading_ref_header(
                        part.content
                    )
                continue
            content = getattr(part, "content", None)
            if isinstance(content, str):
                part.content = strip_inline_citation_tags(content)


# _CITATION_INSTRUCTIONS = """
# Source material (tool results, delegated agent output) begins with a header line:
#   [REF|<key>]
# where <key> is an identifier such as f_1, u_2, or a_3. That first line identifies the block;
# do not treat it as an inline citation in your own prose.

# Only in your **final** message to the user, after a span that draws on such a source,
# add an inline citation using exactly:
#   [REF|<key>]
# If a span uses multiple sources, list every key in one tag, pipe-separated:
#   [REF|key1|key2]
# Do not put [REF|...] tags in assistant messages before the final answer; reserve them for the last user-facing reply.
# """

# _CITATION_INSTRUCTIONS = """
# Cite provided sources using [REF|<key>] inline, placed immediately after
# the specific claim or fact that relies on that source.

# WHEN TO CITE:
# - Specific facts, statistics, numbers, or dates drawn from a source.
# - Direct claims, conclusions, or findings attributable to a source.
# - Technical details, definitions, or domain-specific information from a source.
# - When paraphrasing or summarizing a source's argument or position.
# - If multiple sources support the same claim, list all: [REF|key1|key2].

# WHEN NOT TO CITE:
# - Your own reasoning, synthesis, or logical connectives between ideas.
# - Widely known or common-sense statements (e.g. "the sky is blue").
# - Transitional phrases, introductions, or structural language
#   ("Here's a summary...", "In other words...", "Let's look at...").
# - Conclusions you derive by combining information across sources
#   — unless restating a specific source's conclusion.
# - Repeated references to the same fact within the same paragraph;
#   cite on first mention, then omit for immediate follow-up sentences
#   about the same point.

# RULES:
# - Use ONLY the <key> from each source's [REF|<key>] header.
# - Never invent or hallucinate citation keys.
# - Never use any other citation format (no footnotes, no numbered brackets, no URLs).
# - Aim for precision over volume — one well-placed citation is better than
#   five redundant ones.
# """

_CITATION_INSTRUCTIONS = """
FORMAT:
- Single source: [REF|<key>]
- Multiple sources for one claim: [REF|key1|key2]
- Place the tag immediately after the claim it supports.
- Use ONLY keys from [REF|<key>] headers in provided sources.
- Never invent keys or use any other citation style.

WHEN TO CITE:
- Specific facts, statistics, numbers, or dates from a source.
- Claims, conclusions, or findings attributable to a source.
- Technical details or definitions from a source.
- Paraphrased or summarized arguments from a source.

WHEN NOT TO CITE:
- Your own reasoning, synthesis, or connective language.
- Common-sense or widely known statements.
- Transitional or structural phrases.
- A point you already cited earlier in the same response; cite on first mention only.

CITATION DEPTH:
- Always cite the most specific source available.
- If a subagent result [REF|a_X] contains inline original-source citations [REF|d_X], use the original keys, not a_X.
- Use [REF|a_X] only when a claim has no traceable original key.
- Never combine both: no [REF|a_X|d_Y].

Do not add a "Notes on sources" or summary-of-sources section.
Aim for precision over volume.
"""


@dataclass
class ProvenanceCapability(AbstractCapability):
    """Tracks the full execution provenance of an agent run as a DAG.

    source_tools: names of tools whose return values are raw data sources
    (file readers, API fetchers, etc.). Each invocation gets a unique citation
    key (e.g. "d_1", "d_2") so multiple reads of the same resource are
    tracked independently. Source tool results are wrapped as ``[REF|<key>]`` on
    the first line, with inline ``[REF|...]`` tags stripped from the body so only
    the block header carries the key in tool text. The model is instructed to emit
    inline ``[REF|...]`` only in its final user-facing message.

    When a tool calls a subagent, the subagent's return value is wrapped the same
    way for the parent. These resolve through the shared store
    back to the original DATA_READ nodes for full transitive attribution.

    inject_citation_instructions: when True (default) and source_tools is
    non-empty, citation format instructions are injected automatically via
    get_instructions(). Set to False to manage the prompt yourself.

    After a run, call store.citation_summary() to see what each citation key
    maps to, and use attribute_output() / to_mermaid() etc. for full graph
    analysis.
    """

    source_tools: list[str] = field(default_factory=list)
    agent_name: str = "agent"
    inject_citation_instructions: bool = True

    # Per-run state — populated by for_run(), never set by the caller.
    _store: ProvenanceStore | None = field(default=None, init=False, repr=False)
    _run_id: str = field(default="", init=False, repr=False)
    _run_node_id: str | None = field(default=None, init=False, repr=False)
    _parent_tool_node_id: str | None = field(default=None, init=False, repr=False)
    _ctx_token: Token[Any] | None = field(default=None, init=False, repr=False)
    # Value of _PROVENANCE_CTX before before_run(); used if reset(token) fails
    # because on_run_error / after_run run in a different Context (asyncio).
    _ctx_previous: tuple[ProvenanceStore, str | None] | None = field(
        default=None, init=False, repr=False
    )

    # Tracks the node that the next step should chain from.
    # After parallel tool calls this holds the model-response node so each
    # tool call correctly chains from the same parent.
    _last_sequential_node_id: str | None = field(default=None, init=False, repr=False)

    # Accumulates TOOL_RESULT node IDs from (potentially parallel) tool calls
    # so the next MODEL_REQUEST can draw edges from all of them.
    _pending_tool_result_ids: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def store(self) -> ProvenanceStore:
        if self._store is None:
            raise RuntimeError("store is only available after the agent run starts")
        return self._store

    # ------------------------------------------------------------------
    # Static configuration (called once at agent construction)
    # ------------------------------------------------------------------

    def get_instructions(self) -> str | None:
        if self.inject_citation_instructions:
            return _CITATION_INSTRUCTIONS
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def for_run(self, ctx: RunContext[Any]) -> ProvenanceCapability:
        ctx_value = _PROVENANCE_CTX.get()

        if ctx_value is None:
            store = ProvenanceStore()
            parent_tool_node_id = None
        else:
            store, parent_tool_node_id = ctx_value

        instance = ProvenanceCapability(
            source_tools=self.source_tools,
            agent_name=self.agent_name,
            inject_citation_instructions=self.inject_citation_instructions,
        )
        instance._store = store
        instance._run_id = ctx.run_id or str(uuid.uuid4())
        instance._parent_tool_node_id = parent_tool_node_id
        # Agent keeps the capability instance passed at construction; pydantic-ai
        # runs hooks on the object returned from for_run(). Mirror the store onto
        # this template so callers can still use that reference (e.g. provenance.store
        # after await agent.run()). Not concurrency-safe if the same capability instance
        # is used for overlapping runs.
        self._store = store
        return instance

    async def before_run(self, ctx: RunContext[Any]) -> None:
        store = self._store
        assert store is not None

        run_node = ProvenanceNode.create(
            type=NodeType.AGENT_RUN,
            label=f"Agent: {self.agent_name}",
            agent_name=self.agent_name,
            run_id=self._run_id,
            run_id_value=self._run_id,
        )
        store.add_node(run_node)
        self._run_node_id = run_node.id

        if self._parent_tool_node_id:
            store.add_edge(self._parent_tool_node_id, run_node.id, "delegates_to")

        prompt = ctx.prompt
        if not isinstance(prompt, str):
            prompt = str(prompt) if prompt is not None else ""

        input_node = ProvenanceNode.create(
            type=NodeType.INPUT,
            label="User input",
            agent_name=self.agent_name,
            run_id=self._run_id,
            prompt=prompt,
        )
        store.add_node(input_node)
        store.add_edge(run_node.id, input_node.id, "starts_with")
        self._last_sequential_node_id = input_node.id

        self._ctx_previous = _PROVENANCE_CTX.get()
        self._ctx_token = _PROVENANCE_CTX.set((store, None))

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        store = self._store
        assert store is not None

        output_text = str(result.output) if hasattr(result, "output") else str(result)
        output_node = ProvenanceNode.create(
            type=NodeType.FINAL_OUTPUT,
            label=f"Final output: {self.agent_name}",
            agent_name=self.agent_name,
            run_id=self._run_id,
            output=output_text,
        )
        store.add_node(output_node)

        if self._pending_tool_result_ids:
            for tid in self._pending_tool_result_ids:
                store.add_edge(tid, output_node.id, "produces")
        elif self._last_sequential_node_id:
            store.add_edge(self._last_sequential_node_id, output_node.id, "produces")

        # Register this output so parent agents that receive it as a tool result
        # can wrap it with an [REF:<key>] block and let their
        # LLM emit [REF|<key>] citations that resolve back here.
        store.register_agent_output(output_node.id)

        self._link_citations(output_text, output_node.id, store)

        self._restore_provenance_ctx()

        return result

    async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> Any:
        self._restore_provenance_ctx()
        raise error

    def _restore_provenance_ctx(self) -> None:
        if self._ctx_token is None:
            return
        try:
            _PROVENANCE_CTX.reset(self._ctx_token)
        except ValueError:
            # reset() must run in the same Context as set(); pydantic-ai may
            # deliver on_run_error from another context.
            _PROVENANCE_CTX.set(self._ctx_previous)
        self._ctx_token = None
        self._ctx_previous = None

    # ------------------------------------------------------------------
    # Model hooks
    # ------------------------------------------------------------------

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: Any
    ) -> Any:
        store = self._store
        assert store is not None

        # Scrubbing inline [REF|...] from prior messages before the model is disabled.
        # _strip_inline_refs_from_request_context(request_context)

        request_node = ProvenanceNode.create(
            type=NodeType.MODEL_REQUEST,
            label=f"Model request (step {ctx.run_step})",
            agent_name=self.agent_name,
            run_id=self._run_id,
            step=ctx.run_step,
            message_count=len(request_context.messages),
        )
        store.add_node(request_node)

        if self._pending_tool_result_ids:
            for tid in self._pending_tool_result_ids:
                store.add_edge(tid, request_node.id, "feeds_into")
            self._pending_tool_result_ids = []
        elif self._last_sequential_node_id:
            store.add_edge(self._last_sequential_node_id, request_node.id, "feeds_into")

        self._last_sequential_node_id = request_node.id
        return request_context

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: Any, response: ModelResponse
    ) -> ModelResponse:
        store = self._store
        assert store is not None

        response_node = ProvenanceNode.create(
            type=NodeType.MODEL_RESPONSE,
            label=f"Model response (step {ctx.run_step})",
            agent_name=self.agent_name,
            run_id=self._run_id,
            step=ctx.run_step,
            text=response if response.text else None,
            tool_calls=[tc.tool_name for tc in response.tool_calls],
            model_name=response.model_name,
            input_tokens=response.usage.request_tokens,
            output_tokens=response.usage.response_tokens,
        )
        store.add_node(response_node)
        if self._last_sequential_node_id:
            store.add_edge(self._last_sequential_node_id, response_node.id, "produces")

        if response.text:
            self._link_citations(response.text, response_node.id, store)

        self._last_sequential_node_id = response_node.id
        self._pending_tool_result_ids = []

        return response

    # ------------------------------------------------------------------
    # Tool hooks
    # ------------------------------------------------------------------

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Any,
    ) -> Any:
        store = self._store
        assert store is not None

        is_source = call.tool_name in self.source_tools
        node_type = NodeType.DATA_READ if is_source else NodeType.TOOL_CALL
        label_prefix = "[source] " if is_source else ""
        file_path: str | None = extract_file_path(args) if is_source else None

        call_node = ProvenanceNode.create(
            type=node_type,
            label=f"{label_prefix}Tool: {call.tool_name}",
            agent_name=self.agent_name,
            run_id=self._run_id,
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
            args={k: str(v) for k, v in args.items()},
            file_path=file_path,
        )
        store.add_node(call_node)

        # Register the data source immediately — before the tool runs — so any
        # subagent spawned inside the tool can already resolve citations against it.
        # Each call gets its own unique key, so the same resource read twice produces
        # "d_1" and "d_2" as distinct, independently traceable sources.
        citation_key: str | None = None
        if is_source:
            citation_key = store.register_data_source(call_node.id)

        if self._last_sequential_node_id:
            store.add_edge(self._last_sequential_node_id, call_node.id, "calls")

        token = _PROVENANCE_CTX.set((store, call_node.id))
        try:
            result = await handler(args)
        finally:
            _PROVENANCE_CTX.reset(token)

        # --- Format the result shown to the LLM ---
        #
        # Source tool → [REF|<key>]
        #   LLM emits: [REF|<key>]
        #
        # Any other tool → result unchanged
        returned_result = result
        if is_source and citation_key:
            returned_result = format_cited_content(result, citation_key)
        else:
            subagent_run = self._find_subagent_run(call_node.id, store)
            if subagent_run is not None:
                subagent_key = self._find_citation_key_for_agent(
                    subagent_run.agent_name, subagent_run.run_id, store
                )
                if subagent_key is not None:
                    returned_result = format_cited_content(result, subagent_key)

        result_node = ProvenanceNode.create(
            type=NodeType.TOOL_RESULT,
            label=f"Result: {call.tool_name}",
            agent_name=self.agent_name,
            run_id=self._run_id,
            tool_name=call.tool_name,
            result=str(result),
            formatted=returned_result is not result,
        )
        store.add_node(result_node)
        store.add_edge(call_node.id, result_node.id, "returns")

        self._pending_tool_result_ids.append(result_node.id)

        return returned_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_subagent_run(
        tool_call_node_id: str, store: ProvenanceStore
    ) -> ProvenanceNode | None:
        """Return the AGENT_RUN node spawned by this tool call, if any."""
        for node in store.graph.successors(tool_call_node_id):
            if node.type == NodeType.AGENT_RUN:
                return node
        return None

    @staticmethod
    def _find_citation_key_for_agent(
        agent_name: str, run_id: str, store: ProvenanceStore
    ) -> str | None:
        """Find the citation key assigned to a specific agent run's FINAL_OUTPUT."""
        for key, node_id in store._citation_registry.items():
            node = store.graph.nodes.get(node_id)
            if (
                node is not None
                and node.type == NodeType.FINAL_OUTPUT
                and node.agent_name == agent_name
                and node.run_id == run_id
            ):
                return key
        return None

    def _link_citations(
        self, text: str, target_node_id: str, store: ProvenanceStore
    ) -> None:
        """Parse [REF|<key>] markers and add cited_in edges.

        Resolves each key through the shared citation registry to a node_id,
        then adds a "cited_in" edge: source_node → target_node.

        Because the registry is shared across the entire session, a subagent
        citing ref=d_1 correctly resolves back to a DATA_READ node created
        by a parent agent — giving full transitive attribution across any
        number of agent hops.
        """
        for citation in parse_citations(text):
            for key in citation.refs:
                source_node_id = store.resolve_citation(key)
                if source_node_id and source_node_id != target_node_id:
                    store.add_edge(source_node_id, target_node_id, "cited_in")
