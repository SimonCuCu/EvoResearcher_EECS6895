from pathlib import Path

from evoresearcher.agents.research_agent import ResearchAgent
from evoresearcher.config import AppConfig
from evoresearcher.memory.store import JSONMemoryStore
from evoresearcher.schemas import ConstraintProfile, ResearchBrief


class FakeLLM:
    def __init__(self):
        self.calls = []

    def structured(self, model, *, label, system_prompt, user_prompt, temperature=0.2, model_override=None):
        self.calls.append((label, model_override))
        name = model.__name__
        if name == "RootIdea":
            return model(
                title="Initial MoE Efficiency Hypothesis",
                summary="Use budget-aware routing to reduce wasteful expert dispatch.",
                method_outline="Introduce a routing budget and selective expert activation.",
                evidence_use="Grounded in dispatch cost and communication bottlenecks.",
                risks=["May reduce quality if the budget is too strict."],
            )
        if name == "IdeaReview":
            if "idea-1" in label:
                return model(
                    novelty=7.0,
                    feasibility=6.0,
                    relevance=8.5,
                    clarity=5.5,
                    weakest_dimension="clarity",
                    feedback="Clarify the mechanism and evaluation path.",
                )
            if "idea-2" in label:
                return model(
                    novelty=7.4,
                    feasibility=7.2,
                    relevance=8.1,
                    clarity=7.9,
                    weakest_dimension="feasibility",
                    feedback="Looks strong after clarification.",
                )
            return model(
                novelty=7.1,
                feasibility=6.8,
                relevance=7.9,
                clarity=7.0,
                weakest_dimension="feasibility",
                feedback="Alternative path is decent but somewhat harder to execute.",
            )
        if name == "IdeaExpansion":
            return model(
                refine_weak_dimension_child={
                    "title": "Clarified Budget-Aware Routing",
                    "summary": "Make the routing budget explicit and measurable.",
                    "method_outline": "Define the budgeted dispatch rule and evaluation plan more concretely.",
                    "evidence_use": "Uses the same evidence but turns it into a tighter proposal.",
                    "risks": ["Might still need tuning."],
                    "relation_to_parent": "refine_weak_dimension",
                },
                alternative_direction_child={
                    "title": "Alternative Locality-Aware Dispatch",
                    "summary": "Reduce communication by preferring local experts.",
                    "method_outline": "Bias the router toward low-transfer experts under a soft penalty.",
                    "evidence_use": "Builds on systems evidence around all-to-all cost.",
                    "risks": ["Could hurt balance."],
                    "relation_to_parent": "alternative_direction",
                },
            )
        if name == "PairwiseJudgement":
            return model(
                winner_id="idea-2",
                rationale="The refined child is clearer and easier to validate.",
            )
        if name == "EvidenceSynthesis":
            return model(
                findings=["Communication cost dominates many MoE bottlenecks."],
                tensions=["Efficiency can trade off with quality."],
                opportunities=["Budgeted routing may improve system efficiency."],
            )
        if name == "SearchPlan":
            return model(queries=[])
        raise AssertionError(f"Unexpected model request: {name}")


def build_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        workspace_dir=tmp_path,
        outputs_dir=tmp_path / "outputs",
        memory_dir=tmp_path / "memory",
        author_line="Test",
        deepseek_api_key="test",
        deepseek_model="test-model",
        deepseek_base_url="https://example.com",
        search_enabled=False,
        tree_depth=1,
        branching_factor=2,
    )


def test_review_guided_tree_uses_feedback_structure(tmp_path: Path):
    config = build_config(tmp_path)
    config.ensure_directories()
    brief = ResearchBrief(
        mode="general",
        user_goal="Improve MoE efficiency",
        reframed_goal="Improve MoE efficiency",
        scope="test",
        deliverable="proposal",
        time_cutoff="as of September 2023",
        constraints=ConstraintProfile(),
    )
    llm = FakeLLM()
    agent = ResearchAgent(
        config,
        llm,
        JSONMemoryStore(config.memory_dir / "ideation.json"),
        JSONMemoryStore(config.memory_dir / "proposal.json"),
    )
    result = agent.run(brief)
    assert len(result.idea_tree) == 3
    assert len(result.leaf_ideas) == 2
    root = result.idea_tree[0]
    assert root.relation_to_parent == "root"
    assert root.weakest_dimension == "clarity"
    child_relations = {idea.relation_to_parent for idea in result.leaf_ideas}
    assert child_relations == {"refine_weak_dimension", "alternative_direction"}
    assert result.ranked_ideas[0].title == "Clarified Budget-Aware Routing"
    assert llm.calls
    assert all(model_override == "deepseek-v4-pro" for _, model_override in llm.calls)
