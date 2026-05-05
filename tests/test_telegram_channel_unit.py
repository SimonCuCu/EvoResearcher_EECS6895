import pytest

from evoresearcher.channels.telegram import (
    MODEL_PRESETS,
    TelegramCommandError,
    TelegramConfigError,
    _format_done_message,
    _format_memory_preview,
    _format_selection_question_text,
    is_authorized,
    parse_allowed_user_ids,
    parse_run_command,
    resolve_model_preset,
)
from evoresearcher.runner import RunResult
from evoresearcher.schemas import MemoryEntry


def test_parse_allowed_user_ids_accepts_commas_and_spaces():
    assert parse_allowed_user_ids("123, 456 789") == frozenset({123, 456, 789})


def test_parse_allowed_user_ids_rejects_invalid_values():
    with pytest.raises(TelegramConfigError):
        parse_allowed_user_ids("123,abc")


def test_is_authorized_requires_explicit_allowlist_match():
    allowed = frozenset({123})

    assert is_authorized(123, allowed)
    assert not is_authorized(456, allowed)
    assert not is_authorized(None, allowed)


def test_parse_run_command_defaults_to_general_mode():
    command = parse_run_command("/run investigate MoE routing")

    assert command.mode == "general"
    assert command.goal == "investigate MoE routing"


def test_parse_run_command_accepts_explicit_mode():
    command = parse_run_command("/run ml improve KAN efficiency")

    assert command.mode == "ml"
    assert command.goal == "improve KAN efficiency"


def test_parse_run_command_accepts_bot_mentions():
    command = parse_run_command("/run@EvoResearcherBot general write a proposal")

    assert command.mode == "general"
    assert command.goal == "write a proposal"


def test_parse_run_command_rejects_missing_goal():
    with pytest.raises(TelegramCommandError):
        parse_run_command("/run ml")


def test_flash_model_preset_overrides_all_model_routes():
    preset = resolve_model_preset("flash")

    assert preset is not None
    assert preset.deepseek_model == "deepseek-v4-flash"
    assert preset.deepseek_reasoning_model == "deepseek-v4-flash"
    assert any(item.preset_id == "env" for item in MODEL_PRESETS)


def test_format_memory_preview_includes_example_memory():
    entry = MemoryEntry(
        entry_id="m1",
        kind="ideation",
        summary="Sparse KAN proposal",
        goal="Improve KAN efficiency",
        details="Use adaptive sparse spline grids to reduce dense grid evaluation cost.",
        tags=["kan"],
        created_at="2026-01-01T00:00:00",
    )

    message = _format_memory_preview(
        memory_hits=[entry],
        proposal_hits=[],
        ideation_backend="embedding",
        proposal_backend="embedding",
    )

    assert "Sparse KAN proposal" in message
    assert "adaptive sparse spline grids" in message
    assert "Next I will use these memories" in message


def test_selection_question_text_omits_internal_guidance_labels():
    message = _format_selection_question_text(
        prompt="Which direction should anchor the final report?",
        question_index=1,
        total_questions=2,
        selected_answers=["Candidate preference: idea-2"],
    )

    assert message.startswith("Question 1/2")
    assert "Which direction should anchor the final report?" in message
    assert "Already selected" in message
    assert "Human guidance" not in message
    assert "Human Guidance" not in message
    assert "ML setup" not in message


def test_format_done_message_explains_pdf_fallback(tmp_path):
    result = RunResult(
        run_id="run-1",
        run_dir=tmp_path,
        state={
            "artifacts": {
                "markdown_path": str(tmp_path / "research_report.md"),
                "pdf_warning_path": str(tmp_path / "pdf_render_warning.txt"),
            }
        },
    )

    message = _format_done_message(result)

    assert "PDF rendering did not complete" in message
    assert "Markdown report" in message
