"""Pydantic-AI capability that records a full provenance DAG for every agent run.

Attach a :class:`ProvenanceCapability` to a ``pydantic_ai.Agent`` at construction
time.  The capability hooks into every stage of the agent lifecycle — run start,
model request/response, and tool execution — and builds an attributed directed
acyclic graph that can be inspected after the run via the :attr:`store` property.
"""

from __future__ import annotations

import uuid
from contextvars import Token
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition

from pydantic_ai_provenance.citations import (
    _extract_file_path,
    _format_cited_content,
    parse_citations,
)
from pydantic_ai_provenance.graph import NodeType, ProvenanceNode
from pydantic_ai_provenance.store import _PROVENANCE_CTX, ProvenanceStore
from pydantic_ai_provenance.verification import CitationVerificationReport, verify_citations

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
    _ctx_previous: tuple[ProvenanceStore, str | None] | None = field(default=None, init=False, repr=False)

    # Tracks the node that the next step should chain from.
    # After parallel tool calls this holds the model-response node so each
    # tool call correctly chains from the same parent.
    _last_sequential_node_id: str | None = field(default=None, init=False, repr=False)

    # Accumulates TOOL_RESULT node IDs from (potentially parallel) tool calls
    # so the next MODEL_REQUEST can draw edges from all of them.
    _pending_tool_result_ids: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def store(self) -> ProvenanceStore:
        """The shared :class:`~.store.ProvenanceStore` for the current (or most recent) run.

        Raises:
            RuntimeError: If accessed before the agent run has started (i.e. before
                :meth:`for_run` has been called).
        """
        if self._store is None:
            raise RuntimeError("store is only available after the agent run starts")
        return self._store

    async def verify(
        self,
        text: str,
        *,
        claim_context_chars: int = 720,
        source_max_chars: int = 96_000,
        source_chunk_chars: int = 1_200,
        source_chunk_stride: int = 600,
        source_max_chunks: int = 400,
        min_score: float = 0.3,
        max_keys_per_tag: int = 2,
    ) -> CitationVerificationReport:
        """Verify citation tags in *text* against this capability's shared store.

        Delegates to :func:`~pydantic_ai_provenance.verification.verify_citations`.
        The method is ``async`` so that Step 3 (LLM-based entailment scoring via
        ``await agent.run()``) can be added as an opt-in parameter without a
        breaking API change.

        - **Step 1** strips any ``[REF|…]`` tags whose keys cannot be resolved in
          the store.
        - **Step 2** scores each remaining claim against its cited source via
          TF-IDF cosine similarity and drops keys whose score falls below the
          configured threshold.

        In multi-agent setups every capability shares the same underlying
        :class:`~pydantic_ai_provenance.store.ProvenanceStore`, so it does not
        matter which agent's capability you call this on — the result is identical.
        Use whichever capability reference is most convenient (typically the
        top-level coordinator's).

        Args:
            text: The text containing inline ``[REF|…]`` citation tags to verify.
            claim_context_chars: Maximum characters before each citation tag used
                as the claim for similarity scoring.  Default: ``720``.
            source_max_chars: Maximum source characters fed to the TF-IDF
                vectoriser.  Default: ``96_000``.
            source_chunk_chars: Width of each overlapping source window.  Default: ``1_200``.
            source_chunk_stride: Step size between consecutive source windows.  Default: ``600``.
            source_max_chunks: Maximum number of source windows per citation key.  Default: ``400``.
            min_score: Cosine similarity threshold; keys below this are dropped.
                Default: ``0.3``.
            max_keys_per_tag: Maximum citation keys retained per tag after scoring.
                Default: ``2``.

        Returns:
            A :class:`~pydantic_ai_provenance.verification.CitationVerificationReport`
            with the sanitised text and per-claim similarity records.
        """
        return await verify_citations(
            text,
            self.store,
            claim_context_chars=claim_context_chars,
            source_max_chars=source_max_chars,
            source_chunk_chars=source_chunk_chars,
            source_chunk_stride=source_chunk_stride,
            source_max_chunks=source_max_chunks,
            min_score=min_score,
            max_keys_per_tag=max_keys_per_tag,
        )

    # ------------------------------------------------------------------
    # Static configuration (called once at agent construction)
    # ------------------------------------------------------------------

    def get_instructions(self) -> str | None:
        """Return the citation-format system instructions to inject into the agent prompt.

        When :attr:`inject_citation_instructions` is ``True`` and
        :attr:`source_tools` is non-empty, the model receives formatting rules
        that tell it how and when to emit ``[REF|<key>]`` inline tags.  Returning
        ``None`` means no extra instructions are injected.

        Returns:
            The citation instruction string, or ``None`` if injection is disabled.
        """
        if self.inject_citation_instructions:
            return _CITATION_INSTRUCTIONS
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def for_run(self, ctx: RunContext[Any]) -> ProvenanceCapability:
        """Prepare a per-run capability instance, wiring it into the correct store.

        Called by pydantic-ai before each run begins.  When a provenance context is
        already active (i.e. this agent is a subagent spawned from a tool), the
        existing store and the tool-call node ID that triggered the subagent are
        inherited so subagent nodes are linked back into the parent graph.
        Otherwise a fresh :class:`~.store.ProvenanceStore` is created.

        The template instance (``self``) has its ``_store`` mirrored from the
        per-run instance so that callers who hold a reference to the original
        capability object can still access ``provenance.store`` after
        ``await agent.run()`` returns.  This mirroring is not concurrency-safe
        for overlapping runs on the same capability instance.

        Args:
            ctx: The pydantic-ai run context providing the ``run_id``.

        Returns:
            A freshly constructed :class:`ProvenanceCapability` with per-run state
            initialised but lifecycle hooks not yet fired.
        """
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
        """Record run initialisation nodes and set the provenance context variable.

        Creates an ``AGENT_RUN`` node (linked to the parent tool-call node if this
        is a subagent), followed by an ``INPUT`` node carrying the user prompt.
        The ``_PROVENANCE_CTX`` context variable is then updated so any tools or
        nested agents spawned during the run inherit the same store.

        Args:
            ctx: The pydantic-ai run context providing the user prompt.
        """
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
        """Record the final output node, register its citation key, and restore context.

        Creates a ``FINAL_OUTPUT`` node linked from either the pending tool-result
        nodes (if any) or the last sequential node.  Registers the output in the
        citation registry as an ``a_*`` key so parent agents can cite it.  Parses
        any ``[REF|…]`` tags in the output text and adds ``cited_in`` edges.
        Finally restores ``_PROVENANCE_CTX`` to its pre-run state.

        Args:
            ctx: The pydantic-ai run context (unused directly, required by protocol).
            result: The agent run result; its ``output`` attribute (or its string
                representation) is stored on the output node.

        Returns:
            The unmodified *result* object.
        """
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
        """Restore the provenance context variable and re-raise the error.

        Ensures ``_PROVENANCE_CTX`` is always cleaned up even when the run fails,
        so subsequent runs are not accidentally associated with a stale context.

        Args:
            ctx: The pydantic-ai run context (unused directly).
            error: The exception that caused the run to fail.

        Raises:
            BaseException: Always re-raises *error* unchanged.
        """
        self._restore_provenance_ctx()
        raise error

    def _restore_provenance_ctx(self) -> None:
        """Reset ``_PROVENANCE_CTX`` to its value from before :meth:`before_run`.

        Prefers ``ContextVar.reset(token)`` for clean unwinding.  Falls back to
        ``ContextVar.set(previous_value)`` when ``reset`` raises ``ValueError``,
        which can occur when pydantic-ai delivers :meth:`on_run_error` from a
        different ``asyncio`` task context than the one that called
        :meth:`before_run`.  No-ops if there is no saved token.
        """
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

    async def before_model_request(self, ctx: RunContext[Any], request_context: Any) -> Any:
        """Record a ``MODEL_REQUEST`` node and wire it into the sequential chain.

        Edges are drawn from all pending tool-result nodes (accumulated since the
        last model response) if any exist, otherwise from the last sequential node.
        The pending list is cleared after the edges are added.  The request node
        becomes the new ``_last_sequential_node_id``.

        Args:
            ctx: The pydantic-ai run context, used to read the current step number.
            request_context: The pydantic-ai request context object, returned unchanged.

        Returns:
            The unmodified *request_context*.
        """
        store = self._store
        assert store is not None

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
        """Record a ``MODEL_RESPONSE`` node and parse any inline citations.

        Creates a ``MODEL_RESPONSE`` node linked from the preceding request node.
        If the response contains a text part, :meth:`_link_citations` is called to
        add ``cited_in`` edges for any ``[REF|…]`` tags found in it.  The pending
        tool-result ID list is reset here because a new model turn is starting.

        Args:
            ctx: The pydantic-ai run context, used to read the current step number.
            request_context: The pydantic-ai request context (unused here).
            response: The :class:`~pydantic_ai.messages.ModelResponse` returned by
                the LLM, stored with token usage, model name, and tool-call names.

        Returns:
            The unmodified *response*.
        """
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
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
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
        """Intercept a tool call to record its provenance and optionally wrap its result.

        Creates a ``DATA_READ`` node (for tools in :attr:`source_tools`) or a
        ``TOOL_CALL`` node, executes the underlying tool, then creates a
        ``TOOL_RESULT`` node linked from the call node.

        For source tools the raw result is wrapped with
        :func:`~.citations._format_cited_content` so the model sees a
        ``[REF|<key>]`` block header that it must use when citing facts from the
        result.  For non-source tools, if the tool spawned a subagent (detected by
        looking for an ``AGENT_RUN`` successor), the subagent's output is similarly
        wrapped with its registered ``a_*`` key.

        The ``_PROVENANCE_CTX`` context variable is temporarily set to
        ``(store, call_node.id)`` while the tool runs so any nested agent picks up
        the correct parent link.

        Args:
            ctx: The pydantic-ai run context (unused directly).
            call: The :class:`~pydantic_ai.messages.ToolCallPart` describing the
                tool invocation including its name and call ID.
            tool_def: The :class:`~pydantic_ai.tools.ToolDefinition` for the tool
                being called (unused directly).
            args: Validated keyword arguments that will be forwarded to *handler*.
            handler: Async callable that executes the actual tool logic.

        Returns:
            The (possibly wrapped) tool result string that will be fed back to the
            model.
        """
        store = self._store
        assert store is not None

        is_source = call.tool_name in self.source_tools
        node_type = NodeType.DATA_READ if is_source else NodeType.TOOL_CALL
        label_prefix = "[source] " if is_source else ""
        file_path: str | None = _extract_file_path(args) if is_source else None

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
            returned_result = _format_cited_content(result, citation_key)
        else:
            subagent_run = self._find_subagent_run(call_node.id, store)
            if subagent_run is not None:
                subagent_key = self._find_citation_key_for_agent(subagent_run.agent_name, subagent_run.run_id, store)
                if subagent_key is not None:
                    returned_result = _format_cited_content(result, subagent_key)

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
    def _find_subagent_run(tool_call_node_id: str, store: ProvenanceStore) -> ProvenanceNode | None:
        """Return the AGENT_RUN node spawned by this tool call, if any."""
        for node in store.graph.successors(tool_call_node_id):
            if node.type == NodeType.AGENT_RUN:
                return node
        return None

    @staticmethod
    def _find_citation_key_for_agent(agent_name: str, run_id: str, store: ProvenanceStore) -> str | None:
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

    def _link_citations(self, text: str, target_node_id: str, store: ProvenanceStore) -> None:
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
