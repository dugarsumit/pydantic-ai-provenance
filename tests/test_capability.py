"""Tests for ProvenanceCapability with both sync (agent.run_sync) and async (await agent.run)."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_provenance.capability import ProvenanceCapability
from pydantic_ai_provenance.graph import NodeType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(
    *,
    name: str = "agent",
    source_tools: list[str] | None = None,
    output_text: str = "Done.",
) -> tuple[Agent, ProvenanceCapability]:
    provenance = ProvenanceCapability(agent_name=name, source_tools=source_tools or [])
    agent = Agent(TestModel(custom_output_text=output_text), capabilities=[provenance])
    return agent, provenance


def _expected_node_types() -> set[NodeType]:
    return {NodeType.AGENT_RUN, NodeType.INPUT, NodeType.MODEL_REQUEST, NodeType.MODEL_RESPONSE, NodeType.FINAL_OUTPUT}


# ---------------------------------------------------------------------------
# Single-agent — sync
# ---------------------------------------------------------------------------


def test_sync_run_creates_core_graph_nodes():
    agent, provenance = _agent()
    agent.run_sync("Hello?")
    node_types = {n.type for n in provenance.store.graph.nodes.values()}
    assert _expected_node_types().issubset(node_types)


def test_sync_run_source_tool_registers_data_citation_key():
    agent, provenance = _agent(source_tools=["read_file"])

    @agent.tool_plain
    def read_file(path: str) -> str:
        return "file content"

    agent.run_sync("Read report.txt")
    assert "d_1" in provenance.store.citation_summary()


def test_sync_run_non_source_tool_creates_tool_call_node():
    agent, provenance = _agent()

    @agent.tool_plain
    def compute(x: int) -> int:
        return x * 2

    agent.run_sync("Compute 5")
    store = provenance.store
    tool_call_nodes = [n for n in store.graph.nodes.values() if n.type == NodeType.TOOL_CALL]
    assert len(tool_call_nodes) == 1
    assert tool_call_nodes[0].data["tool_name"] == "compute"


def test_sync_run_citation_link_added_for_inline_ref():
    agent, provenance = _agent(source_tools=["read_file"], output_text="The fox jumped. [REF|d_1]")

    @agent.tool_plain
    def read_file(path: str) -> str:
        return "The quick brown fox."

    agent.run_sync("Read fox.txt")
    cited_edges = [e for e in provenance.store.graph.edges if e.label == "cited_in"]
    assert len(cited_edges) >= 1


def test_sync_run_output_text_stored_on_final_output_node():
    agent, provenance = _agent(output_text="The answer is 42.")
    agent.run_sync("What is the answer?")
    output_nodes = [n for n in provenance.store.graph.nodes.values() if n.type == NodeType.FINAL_OUTPUT]
    assert len(output_nodes) == 1
    assert "42" in output_nodes[0].data.get("output", "")


def test_sync_run_input_prompt_stored_on_input_node():
    agent, provenance = _agent()
    agent.run_sync("My test prompt")
    input_nodes = [n for n in provenance.store.graph.nodes.values() if n.type == NodeType.INPUT]
    assert len(input_nodes) == 1
    assert "My test prompt" in input_nodes[0].data.get("prompt", "")


def test_sync_run_consecutive_runs_produce_fresh_stores():
    agent, provenance = _agent()
    agent.run_sync("First question?")
    store1 = provenance.store
    agent.run_sync("Second question?")
    store2 = provenance.store
    assert store1 is not store2


# ---------------------------------------------------------------------------
# Single-agent — async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_run_creates_core_graph_nodes():
    agent, provenance = _agent()
    await agent.run("Hello?")
    node_types = {n.type for n in provenance.store.graph.nodes.values()}
    assert _expected_node_types().issubset(node_types)


@pytest.mark.asyncio
async def test_async_run_source_tool_registers_data_citation_key():
    agent, provenance = _agent(source_tools=["fetch"])

    @agent.tool_plain
    def fetch(url: str) -> str:
        return "fetched content"

    await agent.run("Fetch example.com")
    assert "d_1" in provenance.store.citation_summary()


@pytest.mark.asyncio
async def test_async_run_citation_link_added_for_inline_ref():
    agent, provenance = _agent(source_tools=["read_file"], output_text="The fox jumped. [REF|d_1]")

    @agent.tool_plain
    def read_file(path: str) -> str:
        return "The quick brown fox."

    await agent.run("Read fox.txt")
    cited_edges = [e for e in provenance.store.graph.edges if e.label == "cited_in"]
    assert len(cited_edges) >= 1


@pytest.mark.asyncio
async def test_async_run_consecutive_runs_produce_fresh_stores():
    agent, provenance = _agent()
    await agent.run("First question?")
    store1 = provenance.store
    await agent.run("Second question?")
    store2 = provenance.store
    assert store1 is not store2


# ---------------------------------------------------------------------------
# Multi-agent — sync parent, async subagent tool
# ---------------------------------------------------------------------------


def test_sync_multi_agent_shared_store():
    """run_sync() on the coordinator; the delegate tool awaits the subagent."""
    research_provenance = ProvenanceCapability(agent_name="researcher", source_tools=["fetch"])
    coordinator_provenance = ProvenanceCapability(agent_name="coordinator")

    research_agent = Agent(TestModel(custom_output_text="Research result."), capabilities=[research_provenance])

    @research_agent.tool_plain
    def fetch(url: str) -> str:
        return "fetched data"

    coordinator_agent = Agent(TestModel(custom_output_text="Summary."), capabilities=[coordinator_provenance])

    @coordinator_agent.tool
    async def delegate(ctx, topic: str) -> str:
        result = await research_agent.run(f"Research {topic}", usage=ctx.usage)
        return result.output

    coordinator_agent.run_sync("Summarise Pydantic AI.")
    store = coordinator_provenance.store
    agent_names = {n.agent_name for n in store.graph.nodes.values()}
    assert "coordinator" in agent_names
    assert "researcher" in agent_names


def test_sync_multi_agent_data_source_key_in_shared_store():
    research_provenance = ProvenanceCapability(agent_name="researcher", source_tools=["fetch"])
    coordinator_provenance = ProvenanceCapability(agent_name="coordinator")

    research_agent = Agent(TestModel(custom_output_text="Research result."), capabilities=[research_provenance])

    @research_agent.tool_plain
    def fetch(url: str) -> str:
        return "fetched data"

    coordinator_agent = Agent(TestModel(custom_output_text="Summary."), capabilities=[coordinator_provenance])

    @coordinator_agent.tool
    async def delegate(ctx, topic: str) -> str:
        result = await research_agent.run(f"Research {topic}", usage=ctx.usage)
        return result.output

    coordinator_agent.run_sync("Summarise Pydantic AI.")
    store = coordinator_provenance.store
    # d_1 is the fetch call in the research subagent; it must be visible in the shared store
    assert "d_1" in store.citation_summary()


# ---------------------------------------------------------------------------
# Multi-agent — async parent, async subagent tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_multi_agent_shared_store():
    research_provenance = ProvenanceCapability(agent_name="researcher", source_tools=["fetch"])
    coordinator_provenance = ProvenanceCapability(agent_name="coordinator")

    research_agent = Agent(TestModel(custom_output_text="Research result."), capabilities=[research_provenance])

    @research_agent.tool_plain
    def fetch(url: str) -> str:
        return "fetched data"

    coordinator_agent = Agent(TestModel(custom_output_text="Summary."), capabilities=[coordinator_provenance])

    @coordinator_agent.tool
    async def delegate(ctx, topic: str) -> str:
        result = await research_agent.run(f"Research {topic}", usage=ctx.usage)
        return result.output

    await coordinator_agent.run("Summarise Pydantic AI.")
    store = coordinator_provenance.store
    agent_names = {n.agent_name for n in store.graph.nodes.values()}
    assert "coordinator" in agent_names
    assert "researcher" in agent_names


@pytest.mark.asyncio
async def test_async_multi_agent_data_source_key_in_shared_store():
    research_provenance = ProvenanceCapability(agent_name="researcher", source_tools=["fetch"])
    coordinator_provenance = ProvenanceCapability(agent_name="coordinator")

    research_agent = Agent(TestModel(custom_output_text="Research result."), capabilities=[research_provenance])

    @research_agent.tool_plain
    def fetch(url: str) -> str:
        return "fetched data"

    coordinator_agent = Agent(TestModel(custom_output_text="Summary."), capabilities=[coordinator_provenance])

    @coordinator_agent.tool
    async def delegate(ctx, topic: str) -> str:
        result = await research_agent.run(f"Research {topic}", usage=ctx.usage)
        return result.output

    await coordinator_agent.run("Summarise Pydantic AI.")
    store = coordinator_provenance.store
    assert "d_1" in store.citation_summary()


@pytest.mark.asyncio
async def test_async_multi_agent_delegates_to_edge_in_graph():
    """The graph should have a delegates_to edge from coordinator's tool call to researcher's AGENT_RUN."""
    research_provenance = ProvenanceCapability(agent_name="researcher", source_tools=["fetch"])
    coordinator_provenance = ProvenanceCapability(agent_name="coordinator")

    research_agent = Agent(TestModel(custom_output_text="Research result."), capabilities=[research_provenance])

    @research_agent.tool_plain
    def fetch(url: str) -> str:
        return "fetched data"

    coordinator_agent = Agent(TestModel(custom_output_text="Summary."), capabilities=[coordinator_provenance])

    @coordinator_agent.tool
    async def delegate(ctx, topic: str) -> str:
        result = await research_agent.run(f"Research {topic}", usage=ctx.usage)
        return result.output

    await coordinator_agent.run("Summarise Pydantic AI.")
    store = coordinator_provenance.store
    delegation_edges = [e for e in store.graph.edges if e.label == "delegates_to"]
    assert len(delegation_edges) >= 1


# ---------------------------------------------------------------------------
# verify() convenience method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_returns_report():
    agent, provenance = _agent(source_tools=["read_file"], output_text="The fox jumped. [REF|d_1]")

    @agent.tool_plain
    def read_file(path: str) -> str:
        return "The quick brown fox jumps over the lazy dog."

    await agent.run("Read fox.txt")
    report = await provenance.verify("The fox jumped. [REF|d_1]")
    assert report.original_text == "The fox jumped. [REF|d_1]"
    assert report.claim_source_similarities != [] or report.text_with_verified_citations is not None


@pytest.mark.asyncio
async def test_verify_multi_agent_same_result_from_either_capability():
    """Both capabilities share the same store — verify() produces identical results."""
    research_provenance = ProvenanceCapability(agent_name="researcher", source_tools=["fetch"])
    coordinator_provenance = ProvenanceCapability(agent_name="coordinator")

    research_agent = Agent(TestModel(custom_output_text="Research result."), capabilities=[research_provenance])

    @research_agent.tool_plain
    def fetch(url: str) -> str:
        return "fetched data"

    coordinator_agent = Agent(TestModel(custom_output_text="Summary."), capabilities=[coordinator_provenance])

    @coordinator_agent.tool
    async def delegate(ctx, topic: str) -> str:
        result = await research_agent.run(f"Research {topic}", usage=ctx.usage)
        return result.output

    await coordinator_agent.run("Summarise Pydantic AI.")

    text = "Some claim. [REF|d_1]"
    report_from_coordinator = await coordinator_provenance.verify(text)
    report_from_researcher = await research_provenance.verify(text)
    assert report_from_coordinator.text_with_verified_citations == report_from_researcher.text_with_verified_citations
