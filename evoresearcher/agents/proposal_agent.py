"""Proposal-writing agent."""

from __future__ import annotations

import re

from pydantic import BaseModel

from evoresearcher.llm import LLMClient
from evoresearcher.schemas import EvidenceSynthesis, ReportSections, ResearchBrief, ResearchIdea, SourceNote


class ProposalAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self,
        *,
        brief: ResearchBrief,
        top_ideas: list[ResearchIdea],
        evidence: EvidenceSynthesis,
        sources: list[SourceNote],
        observer=None,
    ) -> ReportSections:
        if observer is not None:
            observer.phase_log("proposal", f"Writing report from {len(top_ideas)} top ideas.")
        selected_idea = self._select_primary_idea(brief=brief, top_ideas=top_ideas)
        report = self.llm.structured(
            ReportSections,
            label="proposal_sections",
            system_prompt=(
                "You are the proposal agent for EvoResearcher. Write a concise proposal/report "
                "that stays within three pages when rendered in a compact article format. "
                "For ML mode, describe experiments as proposals only; do not claim results. "
                "Stay strictly on the user's topic; do not switch to a different model family or benchmark. "
                "Treat human_clarifications, preferred_idea_id, human_idea_feedback, and report_emphasis in "
                "the brief as binding human-in-the-loop guidance. "
                "If you include equations, use proper LaTeX math delimiters such as $...$ or \\[...\\]. "
                "Never use raw Unicode math symbols like ∈, ≈, Σ, ×, ⊗, π. "
                "Do not emit raw code snippets, class definitions, or unescaped underscores in prose."
            ),
            user_prompt=(
                f"Brief: {brief.model_dump_json(indent=2)}\n"
                f"Human-selected primary idea: {selected_idea.model_dump() if selected_idea else None}\n"
                f"Top ideas: {[idea.model_dump() for idea in top_ideas]}\n"
                f"Evidence synthesis: {evidence.model_dump_json(indent=2)}\n"
                f"Sources: {[source.model_dump() for source in sources]}\n"
                "Structure the report for a deep research audience and include verifiable references. "
                "If a human-selected primary idea is present, anchor the Proposed Direction and report ordering "
                "around that exact idea. Do not assign its idea_id to a different idea. "
                "If you use math, every formula must compile in LaTeX without manual fixes."
            ),
        )
        report = self._sanitize_latex_fragments(report)
        self._validate_latex_fragments(report)
        return report

    def _select_primary_idea(
        self,
        *,
        brief: ResearchBrief,
        top_ideas: list[ResearchIdea],
    ) -> ResearchIdea | None:
        if brief.preferred_idea_id and brief.preferred_idea_id != "synthesize":
            for idea in top_ideas:
                if idea.idea_id == brief.preferred_idea_id:
                    return idea
        return top_ideas[0] if top_ideas else None

    def _sanitize_latex_fragments(self, report: ReportSections) -> ReportSections:
        updates = {}
        text_fields = [
            "abstract",
            "problem_and_goal",
            "evidence_base",
            "proposed_direction",
            "plan_or_analysis",
            "risks_and_limits",
            "conclusion",
        ]
        for field in text_fields:
            updates[field] = self._sanitize_latex_text(getattr(report, field))
        updates["references"] = [self._sanitize_latex_text(reference) for reference in report.references]
        return report.model_copy(update=updates)

    def _sanitize_latex_text(self, text: str) -> str:
        replacements = {
            "∈": "in",
            "≈": "approximately",
            "Σ": "sum",
            "×": "x",
            "⊗": "tensor product",
            "π": "pi",
        }
        for symbol, replacement in replacements.items():
            text = text.replace(symbol, replacement)
        text = re.sub(r"\$(?=\d)", "USD ", text)
        if text.count("$") % 2 != 0:
            text = text.replace("$", "")
        if text.count("{") != text.count("}"):
            text = text.replace("{", "(").replace("}", ")")
        return text

    def _validate_latex_fragments(self, report: ReportSections) -> None:
        joined = "\n".join(
            [
                report.abstract,
                report.problem_and_goal,
                report.evidence_base,
                report.proposed_direction,
                report.plan_or_analysis,
                report.risks_and_limits,
                report.conclusion,
            ]
        )
        if joined.count("$") % 2 != 0:
            raise ValueError("Generated report contains unbalanced inline math delimiters.")
        if joined.count("{") != joined.count("}"):
            raise ValueError("Generated report contains unbalanced braces.")
        forbidden = ("∈", "≈", "Σ", "×", "⊗", "π")
        if any(symbol in joined for symbol in forbidden):
            raise ValueError("Generated report still contains raw Unicode math symbols.")
