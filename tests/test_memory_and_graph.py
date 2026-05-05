import os
from pathlib import Path

import pytest

from evoresearcher.agents.evolution_memory_agent import EvolutionMemoryAgent
from evoresearcher.agents.intake_agent import IntakeAgent
from evoresearcher.agents.proposal_agent import ProposalAgent
from evoresearcher.agents.research_agent import ResearchAgent
from evoresearcher.config import AppConfig
from evoresearcher.llm import LLMClient
from evoresearcher.memory.store import JSONMemoryStore
from evoresearcher.orchestration.graph import build_graph


def build_config(tmp_path: Path) -> AppConfig:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY must be exported for live integration tests.")
    return AppConfig(
        workspace_dir=tmp_path,
        outputs_dir=tmp_path / "outputs",
        memory_dir=tmp_path / "memory",
        author_line="Test",
        deepseek_api_key=api_key,
        deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_base_url=os.environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/chat/completions",
        ),
        deepseek_reasoning_model=os.environ.get("DEEPSEEK_REASONING_MODEL", "deepseek-v4-pro"),
        search_enabled=False,
    )


def test_live_graph(tmp_path: Path):
    config = build_config(tmp_path)
    config.ensure_directories()
    llm = LLMClient(config)
    ideation = JSONMemoryStore(config.memory_dir / "ideation.json")
    proposal = JSONMemoryStore(config.memory_dir / "proposal.json")
    graph = build_graph(
        config=config,
        intake_agent=IntakeAgent(llm),
        research_agent=ResearchAgent(config, llm, ideation, proposal),
        proposal_agent=ProposalAgent(llm),
        ema_agent=EvolutionMemoryAgent(ideation_memory=ideation, proposal_memory=proposal),
    )
    run_dir = config.outputs_dir / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = graph.invoke(
        {
            "run_id": "run-1",
            "run_dir": str(run_dir),
            "mode": "general",
            "goal": "Come up with a novel goal to improve efficiency of MoE",
        }
    )
    assert "report" in state
    assert (run_dir / "research_report.tex").exists()
    assert (run_dir / "research_report.pdf").exists()
    assert (run_dir / "elo_matches.json").exists()
    assert len(state["elo_matches"]) > 0
    assert len(state["memory_updates"]) == 2


def test_second_run_uses_memory(tmp_path: Path):
    config = build_config(tmp_path)
    config.ensure_directories()
    llm = LLMClient(config)
    ideation = JSONMemoryStore(config.memory_dir / "ideation.json")
    proposal = JSONMemoryStore(config.memory_dir / "proposal.json")
    graph = build_graph(
        config=config,
        intake_agent=IntakeAgent(llm),
        research_agent=ResearchAgent(config, llm, ideation, proposal),
        proposal_agent=ProposalAgent(llm),
        ema_agent=EvolutionMemoryAgent(ideation_memory=ideation, proposal_memory=proposal),
    )
    first_state = None
    for run_id in ["run-1", "run-2"]:
        run_dir = config.outputs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state = graph.invoke(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "mode": "general",
                "goal": "Come up with a novel idea to improve efficiency of MoE and generate a proposal",
            }
        )
        if first_state is None:
            first_state = state
    assert len(state["memory_context"]["ideation_hits"]) > 0
    assert len(state["memory_context"]["proposal_hits"]) > 0
    assert state["top_ideas"][0]["total_score"] >= first_state["top_ideas"][0]["total_score"]
