"""LaTeX source generation with real TeX compilation."""

from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import subprocess

from evoresearcher.schemas import ReportSections, ResearchBrief


def render_outputs(
    *,
    run_dir: Path,
    brief: ResearchBrief,
    report: ReportSections,
    sources: list[dict],
    top_ideas: list[dict],
    author_line: str,
) -> dict:
    tex_path = run_dir / "research_report.tex"
    md_path = run_dir / "research_report.md"
    pdf_path = run_dir / "research_report.pdf"
    normalized_report = _normalize_report_sections(report)
    tex_path.write_text(_build_latex(brief=brief, report=normalized_report, author_line=author_line))
    md_path.write_text(_build_markdown(brief=brief, report=report))
    artifacts = {
        "tex_path": str(tex_path),
        "markdown_path": str(md_path),
    }
    try:
        _compile_pdf(tex_path=tex_path, pdf_path=pdf_path)
    except RuntimeError as exc:
        warning_path = run_dir / "pdf_render_warning.txt"
        warning_path.write_text(str(exc))
        artifacts["pdf_warning_path"] = str(warning_path)
    else:
        artifacts["pdf_path"] = str(pdf_path)
    (run_dir / "sources.json").write_text(json.dumps(sources, indent=2))
    (run_dir / "top_ideas.json").write_text(json.dumps(top_ideas, indent=2))
    return artifacts


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    pattern = re.compile("|".join(re.escape(key) for key in replacements))
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def _escape_latex_text_outside_math(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    pattern = re.compile("|".join(re.escape(key) for key in replacements))
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def _normalize_formula_content(text: str) -> str:
    replacements = {
        "∈": r" \in ",
        "≈": r" \approx ",
        "Σ": r"\sum",
        "⊗": r" \otimes ",
        "×": r" \times ",
        "π": r"\pi ",
        "₀": "_0",
        "₁": "_1",
        "₂": "_2",
        "₃": "_3",
        "₄": "_4",
        "₅": "_5",
        "₆": "_6",
        "₇": "_7",
        "₈": "_8",
        "₉": "_9",
        "⁰": "^0",
        "¹": "^1",
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "⁵": "^5",
        "⁶": "^6",
        "⁷": "^7",
        "⁸": "^8",
        "⁹": "^9",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"(?<!\\)\bR\^\{", r"\\mathbb{R}^{", text)
    text = re.sub(r"(?<!\\)\bsin\(", r"\\sin(", text)
    text = re.sub(r"(?<!\\)\bcos\(", r"\\cos(", text)
    text = re.sub(r"(?<!\\)\bexp\(", r"\\exp(", text)
    text = re.sub(r"\s+\*\s+", r" \\cdot ", text)
    return " ".join(text.split())


def _normalize_dims(text: str) -> str:
    return _normalize_formula_content(text)


def _normalize_explicit_formula_patterns(text: str) -> str:
    text = re.sub(
        r"([A-Za-z][A-Za-z0-9]*)\s*∈\s*R\^\{([^}]+)\}",
        lambda match: (
            f"${match.group(1)} \\in \\mathbb{{R}}^{{{_normalize_dims(match.group(2))}}}$"
        ),
        text,
    )
    text = re.sub(
        r"([A-Za-z][A-Za-z0-9]*)\s*≈\s*Σ_\{([^}]+)\}\^([A-Za-z0-9]+)\s*([^.,;\n]+)",
        lambda match: (
            f"${match.group(1)} \\approx \\sum_{{{match.group(2)}}}^{{{match.group(3)}}} "
            f"{_normalize_formula_content(match.group(4))}$"
        ),
        text,
    )
    text = re.sub(
        r"([A-Za-z][A-Za-z0-9]*)\s*=\s*sin\((.*?)\)\s*\+\s*([0-9.]+)\*cos\((.*?)\)\s*\+\s*noise",
        lambda match: (
            f"${match.group(1)} = \\sin({_normalize_formula_content(match.group(2))}) + "
            f"{match.group(3)} \\cdot \\cos({_normalize_formula_content(match.group(4))}) + "
            r"\text{noise}$"
        ),
        text,
    )
    return text


def _normalize_inline_complexity_and_powers(text: str) -> str:
    text = re.sub(
        r"O\(([^)]+)\)",
        lambda match: f"$O({_normalize_formula_content(match.group(1))})$",
        text,
    )
    text = re.sub(
        r"([A-Za-z])\^([0-9]+)",
        lambda match: f"${match.group(1)}^{match.group(2)}$",
        text,
    )
    text = re.sub(
        r"([A-Za-z])_([0-9]+)",
        lambda match: f"${match.group(1)}_{match.group(2)}$",
        text,
    )
    return text


def _normalize_section_text(text: str) -> str:
    text = text.replace("```", "")
    text = text.replace("–", "-").replace("—", "-")
    text = _normalize_explicit_formula_patterns(text)
    text = _normalize_inline_complexity_and_powers(text)
    math_pattern = re.compile(r"(\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\))", re.DOTALL)
    parts = math_pattern.split(text)
    normalized_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        if math_pattern.fullmatch(part):
            if part.startswith("$") and part.endswith("$"):
                inner = part[1:-1]
                normalized_parts.append("$" + _normalize_formula_content(inner) + "$")
            else:
                normalized_parts.append(part)
            continue
        escaped = _escape_latex_text_outside_math(part)
        escaped = escaped.replace("^", r"\textasciicircum{}")
        normalized_parts.append(escaped)
    return "".join(normalized_parts)


def _normalize_report_sections(report: ReportSections) -> ReportSections:
    return report.model_copy(
        update={
            "abstract": _normalize_section_text(report.abstract),
            "problem_and_goal": _normalize_section_text(report.problem_and_goal),
            "evidence_base": _normalize_section_text(report.evidence_base),
            "proposed_direction": _normalize_section_text(report.proposed_direction),
            "plan_or_analysis": _normalize_section_text(report.plan_or_analysis),
            "risks_and_limits": _normalize_section_text(report.risks_and_limits),
            "conclusion": _normalize_section_text(report.conclusion),
            "references": [_escape_latex(ref) for ref in report.references],
        }
    )


def _build_latex(*, brief: ResearchBrief, report: ReportSections, author_line: str) -> str:
    refs = "\n".join([rf"\item {item}" for item in report.references])
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{amsfonts}}
\usepackage{{mathtools}}
\usepackage{{bm}}
\usepackage{{enumitem}}
\usepackage{{hyperref}}
\usepackage{{microtype}}
\usepackage{{parskip}}
\setlength{{\parskip}}{{4pt}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
{{\Large \textbf{{{_escape_latex(report.title)}}}}}\\
{_escape_latex(author_line)}\\
\textit{{Mode: {_escape_latex(brief.mode)} | Time cutoff: {_escape_latex(brief.time_cutoff)}}}
\end{{center}}

\textbf{{Abstract.}} {report.abstract}

\section*{{Problem and Goal}}
{report.problem_and_goal}

\section*{{Evidence Base}}
{report.evidence_base}

\section*{{Proposed Direction}}
{report.proposed_direction}

\section*{{Plan / Analysis}}
{report.plan_or_analysis}

\section*{{Risks and Limits}}
{report.risks_and_limits}

\section*{{Conclusion}}
{report.conclusion}

\section*{{References}}
\begin{{enumerate}}[leftmargin=*]
{refs}
\end{{enumerate}}
\end{{document}}
"""


def _build_markdown(*, brief: ResearchBrief, report: ReportSections) -> str:
    refs = "\n".join(f"- {item}" for item in report.references)
    return f"""# {report.title}

**Mode:** {brief.mode}  
**Time cutoff:** {brief.time_cutoff}

## Abstract
{report.abstract}

## Problem and Goal
{report.problem_and_goal}

## Evidence Base
{report.evidence_base}

## Proposed Direction
{report.proposed_direction}

## Plan / Analysis
{report.plan_or_analysis}

## Risks and Limits
{report.risks_and_limits}

## Conclusion
{report.conclusion}

## References
{refs}
"""


def _compile_pdf(*, tex_path: Path, pdf_path: Path) -> None:
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError("tectonic is required to compile LaTeX reports.")
    log_path = tex_path.with_suffix(".compile.log")
    result = subprocess.run(
        [tectonic, "--keep-logs", "--outdir", str(tex_path.parent), str(tex_path)],
        capture_output=True,
        text=True,
    )
    log_path.write_text(result.stdout + "\n\n" + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"LaTeX compilation failed for {tex_path.name}. See {log_path}."
        )
    generated_pdf = tex_path.with_suffix(".pdf")
    if generated_pdf != pdf_path:
        generated_pdf.replace(pdf_path)
