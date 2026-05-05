"""Top-level LangGraph flow."""

from __future__ import annotations

from pathlib import Path
import json

from langgraph.graph import END, StateGraph

from evoresearcher.agents.evolution_memory_agent import EvolutionMemoryAgent
from evoresearcher.agents.intake_agent import IntakeAgent
from evoresearcher.agents.proposal_agent import ProposalAgent
from evoresearcher.agents.research_agent import ResearchAgent
from evoresearcher.config import AppConfig
from evoresearcher.report.pdf import render_outputs
from evoresearcher.schemas import GraphState, ResearchBrief, ResearchIdea, SourceNote, EvidenceSynthesis, ReportSections


def _select_unique_top_ideas(ranked_ideas: list[ResearchIdea], limit: int = 3) -> list[ResearchIdea]:
    selected: list[ResearchIdea] = []
    seen_titles: set[str] = set()
    for idea in ranked_ideas:
        key = idea.title.strip().lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        selected.append(idea)
        if len(selected) >= limit:
            break
    return selected


def _apply_human_idea_preference(
    *,
    brief: ResearchBrief,
    top_ideas: list[ResearchIdea],
    selection: str,
) -> tuple[ResearchBrief, list[ResearchIdea]]:
    if not selection:
        return brief, top_ideas
    if selection == "__synthesize__":
        return brief.model_copy(update={"preferred_idea_id": "synthesize"}), top_ideas
    matching = [idea for idea in top_ideas if idea.idea_id == selection]
    if not matching:
        return brief.model_copy(update={"human_idea_feedback": selection}), top_ideas
    selected = matching[0]
    reordered = [selected, *[idea for idea in top_ideas if idea.idea_id != selected.idea_id]]
    return brief.model_copy(update={"preferred_idea_id": selected.idea_id}), reordered


def _collect_human_guidance(
    *,
    brief: ResearchBrief,
    top_ideas: list[ResearchIdea],
    observer,
) -> tuple[ResearchBrief, list[ResearchIdea]]:
    if observer is None or not hasattr(observer, "select_option"):
        return brief, top_ideas
    if not top_ideas:
        return brief, top_ideas

    idea_options = [
        {
            "label": f"{idx}. {idea.title[:48]}",
            "value": idea.idea_id,
            "description": f"Use this candidate as the main proposal direction. Score: {idea.total_score:.2f}",
        }
        for idx, idea in enumerate(top_ideas, start=1)
    ]
    idea_options.append(
        {
            "label": "Synthesize",
            "value": "__synthesize__",
            "description": "Let EvoResearcher blend the ranked ideas.",
        }
    )
    selection = observer.select_option(
        title="Human Guidance: Candidate Direction",
        prompt=(
            "I have ranked the candidate ideas. Which direction should anchor the final report? "
            "This choice will reorder the proposal inputs, so the selected idea becomes the primary direction."
        ),
        options=idea_options,
        custom_prompt="Describe the direction or constraint you want the report to follow.",
        question_index=1,
        total_questions=2,
        selected_answers=[],
    )
    brief, top_ideas = _apply_human_idea_preference(
        brief=brief,
        top_ideas=top_ideas,
        selection=selection,
    )

    emphasis = observer.select_option(
        title="Human Guidance: Report Emphasis",
        prompt=(
            "What should the final proposal emphasize? This will be written into the brief and used by the "
            "proposal agent when choosing framing, risk discussion, and evaluation details."
        ),
        options=[
            {
                "label": "Novelty",
                "value": "Emphasize novelty and how this differs from prior work.",
                "description": "Lead with originality and research contribution.",
            },
            {
                "label": "Feasibility",
                "value": "Emphasize feasibility, implementation details, and a practical validation plan.",
                "description": "Make the proposal executable.",
            },
            {
                "label": "Evidence",
                "value": "Emphasize evidence grounding, sources, and assumptions.",
                "description": "Prioritize support from retrieved evidence.",
            },
            {
                "label": "Risks",
                "value": "Emphasize risks, limitations, and mitigation strategies.",
                "description": "Make tradeoffs explicit.",
            },
        ],
        custom_prompt="Describe the emphasis you want in the final report.",
        question_index=2,
        total_questions=2,
        selected_answers=[f"Candidate preference: {selection}"],
    )
    return brief.model_copy(update={"report_emphasis": emphasis}), top_ideas


def build_graph(
    *,
    config: AppConfig,
    intake_agent: IntakeAgent,
    research_agent: ResearchAgent,
    proposal_agent: ProposalAgent,
    ema_agent: EvolutionMemoryAgent,
    observer=None,
):
    graph = StateGraph(GraphState)

    def intake_node(state: GraphState) -> GraphState:
        if observer is not None:
            observer.set_phase("intake", "collecting task brief")
            observer.agent_state("intake", "active", "normalizing user goal")
        brief = intake_agent.run(goal=state["goal"], mode=state["mode"], observer=observer)
        if observer is not None:
            observer.agent_state("intake", "done", "brief ready")
        return {"brief": brief.model_dump()}

    def research_node(state: GraphState) -> GraphState:
        if observer is not None:
            observer.set_phase("research", "retrieving evidence and expanding idea tree")
            observer.agent_state("research", "active", "search + tree + ranking")
        brief = ResearchBrief.model_validate(state["brief"])
        result = research_agent.run(brief, observer=observer)
        top_ideas = _select_unique_top_ideas(result.ranked_ideas, limit=3)
        if observer is not None:
            observer.agent_state("research", "done", f"{len(result.idea_tree)} nodes, {len(result.elo_matches)} Elo matches")
            if hasattr(observer, "ideas_ready"):
                observer.ideas_ready(top_ideas)
        brief, top_ideas = _collect_human_guidance(
            brief=brief,
            top_ideas=top_ideas,
            observer=observer,
        )
        return {
            "brief": brief.model_dump(),
            "sources": [item.model_dump() for item in result.sources],
            "memory_context": result.memory_context,
            "idea_tree": [item.model_dump() for item in result.idea_tree],
            "elo_matches": [item.model_dump() for item in result.elo_matches],
            "top_ideas": [item.model_dump() for item in top_ideas],
            "evidence": result.evidence.model_dump(),
        }

    def proposal_node(state: GraphState) -> GraphState:
        if observer is not None:
            observer.set_phase("proposal", "writing proposal/report")
            observer.agent_state("proposal", "active", "assembling final report")
        brief = ResearchBrief.model_validate(state["brief"])
        top_ideas = [ResearchIdea.model_validate(item) for item in state["top_ideas"]]
        evidence = EvidenceSynthesis.model_validate(state["evidence"])
        sources = [SourceNote.model_validate(item) for item in state["sources"]]
        report = proposal_agent.run(
            brief=brief,
            top_ideas=top_ideas,
            evidence=evidence,
            sources=sources,
            observer=observer,
        )
        if observer is not None:
            observer.agent_state("proposal", "done", "report sections ready")
            if hasattr(observer, "report_ready"):
                observer.report_ready(report)
        return {"report": report.model_dump()}

    def publish_node(state: GraphState) -> GraphState:
        if observer is not None:
            observer.set_phase("publish", "rendering markdown, latex, and pdf outputs")
            observer.agent_state("publish", "active", "writing artifacts")
        run_dir = Path(state["run_dir"])
        brief = ResearchBrief.model_validate(state["brief"])
        report = ReportSections.model_validate(state["report"])
        artifacts = render_outputs(
            run_dir=run_dir,
            brief=brief,
            report=report,
            sources=state["sources"],
            top_ideas=state["top_ideas"],
            author_line=config.author_line,
        )
        for label, path in artifacts.items():
            if observer is not None:
                observer.artifact(label, Path(path))
        (run_dir / "idea_tree.json").write_text(json.dumps(state["idea_tree"], indent=2))
        (run_dir / "elo_matches.json").write_text(json.dumps(state["elo_matches"], indent=2))
        (run_dir / "evidence.json").write_text(json.dumps(state["evidence"], indent=2))
        (run_dir / "memory_context.json").write_text(json.dumps(state["memory_context"], indent=2))
        if observer is not None:
            observer.artifact("elo_matches", run_dir / "elo_matches.json")
            observer.artifact("memory_context", run_dir / "memory_context.json")
            observer.agent_state("publish", "done", "artifacts written")
        return {"artifacts": artifacts}

    def ema_node(state: GraphState) -> GraphState:
        if observer is not None:
            observer.set_phase("memory", "updating evolution memories")
            observer.agent_state("ema", "active", "writing back learned patterns")
        brief = ResearchBrief.model_validate(state["brief"])
        top_idea = ResearchIdea.model_validate(state["top_ideas"][0])
        evidence = EvidenceSynthesis.model_validate(state["evidence"])
        report = ReportSections.model_validate(state["report"])
        updates = ema_agent.run(
            brief=brief,
            top_idea=top_idea,
            evidence=evidence,
            report=report,
            observer=observer,
        )
        (Path(state["run_dir"]) / "memory_updates.json").write_text(json.dumps(updates, indent=2))
        if observer is not None:
            observer.artifact("memory_updates", Path(state["run_dir"]) / "memory_updates.json")
            observer.agent_state("ema", "done", "memories updated")
        return {"memory_updates": updates}

    graph.add_node("intake", intake_node)
    graph.add_node("research", research_node)
    graph.add_node("proposal", proposal_node)
    graph.add_node("publish", publish_node)
    graph.add_node("ema", ema_node)
    graph.set_entry_point("intake")
    graph.add_edge("intake", "research")
    graph.add_edge("research", "proposal")
    graph.add_edge("proposal", "publish")
    graph.add_edge("publish", "ema")
    graph.add_edge("ema", END)
    return graph.compile()
