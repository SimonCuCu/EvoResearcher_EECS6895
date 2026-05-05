"""Reusable EvoResearcher run orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from evoresearcher.agents.evolution_memory_agent import EvolutionMemoryAgent
from evoresearcher.agents.intake_agent import IntakeAgent
from evoresearcher.agents.proposal_agent import ProposalAgent
from evoresearcher.agents.research_agent import ResearchAgent
from evoresearcher.config import load_config
from evoresearcher.llm import LLMClient
from evoresearcher.memory.store import JSONMemoryStore
from evoresearcher.orchestration.graph import build_graph
from evoresearcher.schemas import ModeName


@dataclass(slots=True)
class RunOptions:
    workspace_dir: str | None = None
    search_enabled: bool = True
    tree_depth: int = 2
    branching_factor: int = 2
    max_sources: int = 6
    deepseek_model: str | None = None
    deepseek_reasoning_model: str | None = None


@dataclass(slots=True)
class RunResult:
    run_id: str
    run_dir: Path
    state: dict[str, Any]


def run_research(
    *,
    goal: str,
    mode: ModeName = "general",
    options: RunOptions | None = None,
    observer=None,
) -> RunResult:
    """Run the full research pipeline and return the final state."""

    goal = goal.strip()
    if not goal:
        raise ValueError("A non-empty goal is required.")

    opts = options or RunOptions()
    config = load_config(
        workspace_dir=opts.workspace_dir,
        search_enabled=opts.search_enabled,
        tree_depth=opts.tree_depth,
        branching_factor=opts.branching_factor,
        max_sources=opts.max_sources,
        deepseek_model=opts.deepseek_model,
        deepseek_reasoning_model=opts.deepseek_reasoning_model,
    )
    run_id = config.make_run_id(goal)
    run_dir = config.outputs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMClient(config)
    ideation_memory = JSONMemoryStore(config.memory_dir / "ideation_memory.json")
    proposal_memory = JSONMemoryStore(config.memory_dir / "proposal_memory.json")
    intake_agent = IntakeAgent(llm)
    research_agent = ResearchAgent(config, llm, ideation_memory, proposal_memory)
    proposal_agent = ProposalAgent(llm)
    ema_agent = EvolutionMemoryAgent(
        ideation_memory=ideation_memory,
        proposal_memory=proposal_memory,
    )

    if observer is not None and hasattr(observer, "start_run"):
        observer.start_run(
            run_id=run_id,
            mode=mode,
            goal=goal,
            model_name=f"{config.deepseek_model} (research reasoning: {config.deepseek_reasoning_model})",
            provider="deepseek",
            workspace_dir=config.workspace_dir,
        )

    app = build_graph(
        config=config,
        intake_agent=intake_agent,
        research_agent=research_agent,
        proposal_agent=proposal_agent,
        ema_agent=ema_agent,
        observer=observer,
    )
    model_routing = {
        "provider": "deepseek",
        "default_model": config.deepseek_model,
        "research_reasoning_model": config.deepseek_reasoning_model,
        "research_reasoning_labels": [
            "research_search_plan",
            "research_root_idea",
            "research_expansion_depth_*",
            "research_review_*",
            "elo_judge_*",
            "research_evidence_synthesis",
        ],
        "default_model_labels": [
            "intake_*",
            "ml_intake_*",
            "proposal_sections",
        ],
    }
    state = app.invoke(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "mode": mode,
            "goal": goal,
            "model_routing": model_routing,
        }
    )
    state["model_routing"] = model_routing
    model_routing_path = run_dir / "model_routing.json"
    model_routing_path.write_text(json.dumps(model_routing, indent=2))
    state.setdefault("artifacts", {})["model_routing_path"] = str(model_routing_path)
    (run_dir / "run_summary.json").write_text(json.dumps(state, indent=2))
    if observer is not None and hasattr(observer, "artifact"):
        observer.artifact("run_summary", run_dir / "run_summary.json")
    if observer is not None and hasattr(observer, "finish"):
        observer.finish("run completed")
    return RunResult(run_id=run_id, run_dir=run_dir, state=state)
