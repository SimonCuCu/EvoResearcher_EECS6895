"""Interactive intake agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from rich.prompt import Prompt

from evoresearcher.llm import LLMClient
from evoresearcher.schemas import ConstraintProfile, ResearchBrief


class BriefNormalization(BaseModel):
    reframed_goal: str
    scope: str
    deliverable: str
    time_cutoff: str
    key_questions: list[str] = Field(default_factory=list)


class MLIntakeOption(BaseModel):
    label: str
    value: str
    description: str = ""


class MLIntakeQuestion(BaseModel):
    field_name: Literal[
        "model_scale",
        "dataset_scope",
        "compute_budget",
        "time_budget",
        "extra_notes",
    ]
    title: str
    prompt: str
    options: list[MLIntakeOption] = Field(default_factory=list)


class MLIntakeQuestionnaire(BaseModel):
    questions: list[MLIntakeQuestion] = Field(default_factory=list)


class ClarifyingQuestion(BaseModel):
    title: str
    prompt: str
    options: list[MLIntakeOption] = Field(default_factory=list)


class ClarifyingQuestionnaire(BaseModel):
    questions: list[ClarifyingQuestion] = Field(default_factory=list)


class IntakeAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, *, goal: str, mode: str, observer=None) -> ResearchBrief:
        self._current_goal = goal
        constraints = ConstraintProfile()
        if mode == "ml":
            constraints = self._collect_ml_constraints(observer=observer)
        human_clarifications: list[str] = []
        if self._should_collect_clarifications(goal):
            human_clarifications = self._collect_clarifications(
                goal=goal,
                mode=mode,
                constraints=constraints,
                observer=observer,
            )
        if observer is not None:
            observer.phase_log("intake", "Normalizing goal into a structured research brief.")
        normalized = self.llm.structured(
            BriefNormalization,
            label="intake_brief_normalization",
            system_prompt=(
                "You are the intake agent for EvoResearcher. Rewrite the user's request into a "
                "precise deep-research brief suitable for a proposal-generation pipeline."
            ),
            user_prompt=(
                f"Mode: {mode}\n"
                f"User goal: {goal}\n"
                f"Constraints: {constraints.model_dump_json(indent=2)}\n"
                f"Human clarifications: {human_clarifications}\n"
                "Preserve any explicit time cutoff such as 'as of September 2023'."
            ),
        )
        return ResearchBrief(
            mode=mode,
            user_goal=goal,
            reframed_goal=normalized.reframed_goal,
            scope=normalized.scope,
            deliverable=normalized.deliverable,
            time_cutoff=normalized.time_cutoff,
            key_questions=normalized.key_questions,
            constraints=constraints,
            human_clarifications=human_clarifications,
        )

    def _should_collect_clarifications(self, goal: str) -> bool:
        lowered = goal.lower()
        triggers = [
            "ask me clarifying",
            "ask clarifying",
            "clarifying question",
            "clarification question",
            "先问",
            "澄清问题",
            "先澄清",
        ]
        return any(trigger in lowered for trigger in triggers)

    def _collect_clarifications(
        self,
        *,
        goal: str,
        mode: str,
        constraints: ConstraintProfile,
        observer=None,
    ) -> list[str]:
        if observer is not None:
            observer.phase_log("intake", "Generating clarifying questions before normalizing the brief.")
        questionnaire = self.llm.structured(
            ClarifyingQuestionnaire,
            label="intake_clarifying_questionnaire",
            system_prompt=(
                "You are the human-in-the-loop intake designer for EvoResearcher. Generate two or three "
                "high-impact clarification questions. Each answer must materially change scope, evidence "
                "selection, feasibility constraints, or report emphasis."
            ),
            user_prompt=(
                f"Mode: {mode}\n"
                f"User goal: {goal}\n"
                f"Existing constraints: {constraints.model_dump_json(indent=2)}\n\n"
                "Return 2-3 questions. Each question must have exactly 3 concise predefined options. "
                "Do not include a custom option; the UI adds it."
            ),
        )
        questions = self._normalize_clarifying_questions(questionnaire)
        answers: list[str] = []
        for idx, question in enumerate(questions, start=1):
            title = f"Clarification Q{idx}: {question.title}"
            if observer is not None and hasattr(observer, "select_option"):
                answer = observer.select_option(
                    title=title,
                    prompt=question.prompt,
                    options=[option.model_dump() for option in question.options],
                    custom_prompt="Type your clarification.",
                    question_index=idx,
                    total_questions=len(questions),
                    selected_answers=answers,
                )
            else:
                answer = self._fallback_clarifying_prompt(question, title)
            answers.append(f"{question.title}: {answer}")
        return answers

    def _normalize_clarifying_questions(
        self,
        questionnaire: ClarifyingQuestionnaire,
    ) -> list[ClarifyingQuestion]:
        questions = questionnaire.questions[:3]
        if len(questions) < 2:
            raise ValueError("Clarifying questionnaire returned too few questions.")
        normalized: list[ClarifyingQuestion] = []
        for idx, question in enumerate(questions, start=1):
            options = question.options[:3]
            if len(options) < 2:
                raise ValueError(f"Clarifying question {idx} returned too few options.")
            normalized.append(question.model_copy(update={"options": options}))
        return normalized

    def _fallback_clarifying_prompt(self, question: ClarifyingQuestion, title: str) -> str:
        options = list(question.options)
        options.append(
            MLIntakeOption(
                label="Custom",
                value="__custom__",
                description="Type your own answer.",
            )
        )
        display = "\n".join(
            [f"{idx}. {option.label} - {option.description}" for idx, option in enumerate(options, start=1)]
        )
        choice = Prompt.ask(f"[bold cyan]{title}[/bold cyan]\n{question.prompt}\n{display}", default="1")
        try:
            selected = options[max(0, min(len(options) - 1, int(choice) - 1))]
        except ValueError:
            selected = options[-1]
        if selected.value == "__custom__":
            return Prompt.ask("Type your clarification", default="")
        return selected.value

    def _collect_ml_constraints(self, observer=None) -> ConstraintProfile:
        if observer is not None:
            observer.phase_log("intake", "Generating ML intake questions and options.")
        questionnaire = self.llm.structured(
            MLIntakeQuestionnaire,
            label="ml_intake_questionnaire",
            system_prompt=(
                "You are the ML intake designer for EvoResearcher. Generate exactly five concise but useful "
                "intake questions for an ML research proposal workflow. Each question must target one of these "
                "fields exactly once: model_scale, dataset_scope, compute_budget, time_budget, extra_notes. "
                "For each question, return 3 short predefined options. Do not include a custom option; the UI adds it."
            ),
            user_prompt=(
                "User is asking for an ML research proposal. Tailor the questions to the specific goal below.\n"
                "Goal:\n"
                f"{getattr(self, '_current_goal', '')}\n\n"
                "Requirements:\n"
                "- Options should be realistic and mutually distinct.\n"
                "- Keep labels short enough to fit in terminal buttons.\n"
                "- Put any framework preference question under extra_notes.\n"
                "- Return exactly five questions, one per required field."
            ),
        )
        answers_by_field: dict[str, str] = {}
        questions = self._normalize_questionnaire(questionnaire)
        selected_answers: list[str] = []
        for idx, question in enumerate(questions, start=1):
            title = f"ML Intake Q{idx}: {question.title}"
            custom_prompt = self._custom_prompt_for_field(question.field_name)
            if observer is not None and hasattr(observer, "select_option"):
                answer = observer.select_option(
                    title=title,
                    prompt=question.prompt,
                    options=[option.model_dump() for option in question.options],
                    custom_prompt=custom_prompt,
                    question_index=idx,
                    total_questions=len(questions),
                    selected_answers=selected_answers,
                )
            else:
                answer = self._fallback_prompt(question, title, custom_prompt)
            answers_by_field[question.field_name] = answer
            selected_answers.append(f"{question.title}: {answer}")
        return ConstraintProfile(
            model_scale=answers_by_field["model_scale"],
            dataset_scope=answers_by_field["dataset_scope"],
            compute_budget=answers_by_field["compute_budget"],
            time_budget=answers_by_field["time_budget"],
            extra_notes=answers_by_field["extra_notes"],
        )

    def _normalize_questionnaire(
        self,
        questionnaire: MLIntakeQuestionnaire,
    ) -> list[MLIntakeQuestion]:
        expected_order = [
            "model_scale",
            "dataset_scope",
            "compute_budget",
            "time_budget",
            "extra_notes",
        ]
        by_field = {question.field_name: question for question in questionnaire.questions}
        missing = [field for field in expected_order if field not in by_field]
        if missing:
            raise ValueError(f"ML intake questionnaire missing fields: {missing}")
        normalized: list[MLIntakeQuestion] = []
        for field in expected_order:
            question = by_field[field]
            options = question.options[:3]
            if len(options) < 2:
                raise ValueError(f"ML intake question '{field}' returned too few options.")
            normalized.append(
                question.model_copy(
                    update={
                        "options": options,
                    }
                )
            )
        return normalized

    def _fallback_prompt(
        self,
        question: MLIntakeQuestion,
        title: str,
        custom_prompt: str,
    ) -> str:
        options = list(question.options)
        options.append(
            MLIntakeOption(
                label="Custom",
                value="__custom__",
                description="Type your own answer.",
            )
        )
        display = "\n".join(
            [f"{idx}. {option.label} - {option.description}" for idx, option in enumerate(options, start=1)]
        )
        choice = Prompt.ask(f"[bold cyan]{title}[/bold cyan]\n{question.prompt}\n{display}", default="1")
        try:
            selected = options[max(0, min(len(options) - 1, int(choice) - 1))]
        except ValueError:
            selected = options[-1]
        if selected.value == "__custom__":
            return Prompt.ask(custom_prompt, default="")
        return selected.value

    def _custom_prompt_for_field(self, field_name: str) -> str:
        prompts = {
            "model_scale": "Enter your target model or MoE scale",
            "dataset_scope": "Enter your dataset scope or benchmark preference",
            "compute_budget": "Enter your local compute budget",
            "time_budget": "Enter your time budget",
            "extra_notes": "Enter your framework preference or extra notes",
        }
        return prompts[field_name]
