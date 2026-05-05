from evoresearcher.agents.intake_agent import IntakeAgent
from evoresearcher.orchestration.graph import _apply_human_idea_preference
from evoresearcher.schemas import ResearchBrief, ResearchIdea


class FakeClarifyingLLM:
    def __init__(self):
        self.model_overrides = []

    def structured(self, model, *, label, system_prompt, user_prompt, temperature=0.2, model_override=None):
        self.model_overrides.append(model_override)
        name = model.__name__
        if name == "ClarifyingQuestionnaire":
            return model(
                questions=[
                    {
                        "title": "Scope",
                        "prompt": "Which market should the research focus on?",
                        "options": [
                            {"label": "China", "value": "Focus on China", "description": "China market"},
                            {"label": "US", "value": "Focus on US", "description": "US market"},
                            {"label": "Global", "value": "Global comparison", "description": "Global market"},
                        ],
                    },
                    {
                        "title": "Output",
                        "prompt": "What output should be prioritized?",
                        "options": [
                            {"label": "Proposal", "value": "Prioritize a proposal", "description": "Research plan"},
                            {"label": "Survey", "value": "Prioritize a survey", "description": "Literature review"},
                            {"label": "Data", "value": "Prioritize data", "description": "Empirical plan"},
                        ],
                    },
                ]
            )
        if name == "BriefNormalization":
            assert "Focus on China" in user_prompt
            assert "Prioritize a proposal" in user_prompt
            return model(
                reframed_goal="Research green bonds in China",
                scope="China green bond market",
                deliverable="Proposal",
                time_cutoff="current",
                key_questions=["What explains performance?"],
            )
        raise AssertionError(f"Unexpected model request: {name}")


class FirstOptionObserver:
    def __init__(self):
        self.prompts = []

    def phase_log(self, phase: str, message: str) -> None:
        pass

    def select_option(
        self,
        *,
        title,
        prompt,
        options,
        custom_prompt,
        question_index,
        total_questions,
        selected_answers,
    ):
        self.prompts.append((title, prompt))
        return options[0]["value"]


def make_idea(idea_id: str, title: str, score: float) -> ResearchIdea:
    return ResearchIdea(
        idea_id=idea_id,
        title=title,
        summary=f"Summary for {title}",
        method_outline=f"Method for {title}",
        evidence_use="Evidence",
        total_score=score,
    )


def test_intake_clarifying_answers_are_added_to_brief():
    observer = FirstOptionObserver()
    llm = FakeClarifyingLLM()
    agent = IntakeAgent(llm)

    brief = agent.run(
        goal="Ask me clarifying questions before researching green bonds.",
        mode="general",
        observer=observer,
    )

    assert len(observer.prompts) == 2
    assert brief.human_clarifications == [
        "Scope: Focus on China",
        "Output: Prioritize a proposal",
    ]
    assert brief.scope == "China green bond market"
    assert llm.model_overrides == [None, None]


def test_human_idea_preference_reorders_top_ideas():
    brief = ResearchBrief(
        mode="general",
        user_goal="goal",
        reframed_goal="goal",
        scope="scope",
        deliverable="report",
        time_cutoff="none",
    )
    ideas = [
        make_idea("idea-1", "First", 7.0),
        make_idea("idea-2", "Second", 8.0),
        make_idea("idea-3", "Third", 6.0),
    ]

    updated_brief, reordered = _apply_human_idea_preference(
        brief=brief,
        top_ideas=ideas,
        selection="idea-2",
    )

    assert updated_brief.preferred_idea_id == "idea-2"
    assert [idea.idea_id for idea in reordered] == ["idea-2", "idea-1", "idea-3"]


def test_custom_idea_feedback_is_preserved_without_reordering():
    brief = ResearchBrief(
        mode="general",
        user_goal="goal",
        reframed_goal="goal",
        scope="scope",
        deliverable="report",
        time_cutoff="none",
    )
    ideas = [make_idea("idea-1", "First", 7.0)]

    updated_brief, reordered = _apply_human_idea_preference(
        brief=brief,
        top_ideas=ideas,
        selection="Focus more on policy feasibility",
    )

    assert updated_brief.human_idea_feedback == "Focus more on policy feasibility"
    assert reordered == ideas
