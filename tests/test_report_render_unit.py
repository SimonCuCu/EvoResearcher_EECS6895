from pathlib import Path

import pytest

from evoresearcher.report import pdf
from evoresearcher.schemas import ReportSections, ResearchBrief


def test_render_outputs_keeps_markdown_when_pdf_compile_fails(monkeypatch, tmp_path: Path):
    brief = ResearchBrief(
        mode="general",
        user_goal="goal",
        reframed_goal="goal",
        scope="scope",
        deliverable="report",
        time_cutoff="none",
    )
    report = ReportSections(
        title="Test Report",
        abstract="Abstract",
        problem_and_goal="Problem",
        evidence_base="Evidence",
        proposed_direction="Direction",
        plan_or_analysis="Plan",
        risks_and_limits="Risks",
        conclusion="Conclusion",
        references=["Reference"],
    )

    def fail_compile(*, tex_path: Path, pdf_path: Path) -> None:
        raise RuntimeError("LaTeX compilation failed")

    monkeypatch.setattr(pdf, "_compile_pdf", fail_compile)

    artifacts = pdf.render_outputs(
        run_dir=tmp_path,
        brief=brief,
        report=report,
        sources=[],
        top_ideas=[],
        author_line="Test",
    )

    assert (tmp_path / "research_report.md").exists()
    assert (tmp_path / "research_report.tex").exists()
    assert (tmp_path / "pdf_render_warning.txt").exists()
    assert "markdown_path" in artifacts
    assert "pdf_path" not in artifacts
    assert "pdf_warning_path" in artifacts
