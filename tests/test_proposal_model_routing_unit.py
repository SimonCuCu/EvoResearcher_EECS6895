from evoresearcher.agents.proposal_agent import ProposalAgent
from evoresearcher.schemas import EvidenceSynthesis, ReportSections, ResearchBrief, ResearchIdea


class FakeProposalLLM:
    def __init__(self, *, abstract="Abstract"):
        self.model_overrides = []
        self.user_prompts = []
        self.abstract = abstract

    def structured(self, model, *, label, system_prompt, user_prompt, temperature=0.2, model_override=None):
        self.model_overrides.append(model_override)
        self.user_prompts.append(user_prompt)
        return model(
            title="Report",
            abstract=self.abstract,
            problem_and_goal="Problem",
            evidence_base="Evidence",
            proposed_direction="Direction",
            plan_or_analysis="Plan",
            risks_and_limits="Risks",
            conclusion="Conclusion",
            references=["Reference"],
        )


def test_proposal_agent_uses_default_model_route():
    llm = FakeProposalLLM()
    agent = ProposalAgent(llm)
    brief = ResearchBrief(
        mode="general",
        user_goal="goal",
        reframed_goal="goal",
        scope="scope",
        deliverable="report",
        time_cutoff="none",
    )
    idea = ResearchIdea(
        idea_id="idea-1",
        title="Idea",
        summary="Summary",
        method_outline="Method",
        evidence_use="Evidence",
    )

    agent.run(
        brief=brief,
        top_ideas=[idea],
        evidence=EvidenceSynthesis(),
        sources=[],
    )

    assert llm.model_overrides == [None]


def test_proposal_agent_sanitizes_report_text_before_validation():
    llm = FakeProposalLLM(abstract="Fee was $0.66 and stray $ marker with x ∈ X and {bad.")
    agent = ProposalAgent(llm)
    brief = ResearchBrief(
        mode="general",
        user_goal="goal",
        reframed_goal="goal",
        scope="scope",
        deliverable="report",
        time_cutoff="none",
    )
    idea = ResearchIdea(
        idea_id="idea-1",
        title="Idea",
        summary="Summary",
        method_outline="Method",
        evidence_use="Evidence",
    )

    report = agent.run(
        brief=brief,
        top_ideas=[idea],
        evidence=EvidenceSynthesis(),
        sources=[],
    )

    assert "USD 0.66" in report.abstract
    assert "$" not in report.abstract
    assert "in X" in report.abstract
    assert "{" not in report.abstract


def test_proposal_prompt_includes_human_selected_primary_idea():
    llm = FakeProposalLLM()
    agent = ProposalAgent(llm)
    brief = ResearchBrief(
        mode="general",
        user_goal="goal",
        reframed_goal="goal",
        scope="scope",
        deliverable="report",
        time_cutoff="none",
        preferred_idea_id="idea-2",
    )
    ideas = [
        ResearchIdea(
            idea_id="idea-1",
            title="First idea",
            summary="Summary",
            method_outline="Method",
            evidence_use="Evidence",
        ),
        ResearchIdea(
            idea_id="idea-2",
            title="Human chosen idea",
            summary="Chosen summary",
            method_outline="Chosen method",
            evidence_use="Chosen evidence",
        ),
    ]

    agent.run(
        brief=brief,
        top_ideas=ideas,
        evidence=EvidenceSynthesis(),
        sources=[],
    )

    assert "Human-selected primary idea" in llm.user_prompts[0]
    assert "Human chosen idea" in llm.user_prompts[0]
    assert "Do not assign its idea_id to a different idea" in llm.user_prompts[0]
