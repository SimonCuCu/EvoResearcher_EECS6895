"""Telegram channel for running EvoResearcher from a phone."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from dataclasses import dataclass
from itertools import count
from pathlib import Path
import os
import re
import threading
import time
from typing import Any

from dotenv import load_dotenv

from evoresearcher.runner import RunOptions, RunResult, run_research
from evoresearcher.schemas import MemoryEntry, ModeName, ReportSections, ResearchIdea


COMMANDS = """Use /start to choose a task type and model profile.

While a run is active, you can also use:
/status
/last
/cancel"""

MODE_CALLBACK_PREFIX = "mode:"
ML_CALLBACK_PREFIX = "ml:"
ML_CUSTOM_CALLBACK_PREFIX = "mlc:"
MODEL_CALLBACK_PREFIX = "model:"


class TelegramConfigError(RuntimeError):
    """Raised when Telegram channel configuration is invalid."""


class TelegramCommandError(ValueError):
    """Raised when a Telegram command cannot be parsed."""


@dataclass(frozen=True, slots=True)
class TelegramSettings:
    bot_token: str
    allowed_user_ids: frozenset[int]
    workspace_dir: str | None = None
    search_enabled: bool = True
    tree_depth: int = 2
    branching_factor: int = 2
    max_sources: int = 6


@dataclass(frozen=True, slots=True)
class ModelPreset:
    preset_id: str
    label: str
    description: str
    deepseek_model: str | None = None
    deepseek_reasoning_model: str | None = None


@dataclass(frozen=True, slots=True)
class RunCommand:
    mode: ModeName
    goal: str
    deepseek_model: str | None = None
    deepseek_reasoning_model: str | None = None
    model_label: str = "Default (.env)"


MODEL_PRESETS = (
    ModelPreset(
        preset_id="env",
        label="Default (.env)",
        description="Use DEEPSEEK_MODEL and DEEPSEEK_REASONING_MODEL from .env.",
    ),
    ModelPreset(
        preset_id="flash",
        label="Flash",
        description="Use deepseek-v4-flash for every LLM call.",
        deepseek_model="deepseek-v4-flash",
        deepseek_reasoning_model="deepseek-v4-flash",
    ),
    ModelPreset(
        preset_id="pro-reasoning",
        label="Pro reasoning",
        description="Use default model for intake/proposal and deepseek-v4-pro for research reasoning.",
        deepseek_reasoning_model="deepseek-v4-pro",
    ),
)


def resolve_model_preset(preset_id: str) -> ModelPreset | None:
    return next((preset for preset in MODEL_PRESETS if preset.preset_id == preset_id), None)


@dataclass(slots=True)
class PendingSelection:
    chat_id: int
    request_id: int
    title: str
    options: list[dict]
    future: Future[str]


def parse_allowed_user_ids(value: str) -> frozenset[int]:
    if not value.strip():
        return frozenset()
    user_ids: set[int] = set()
    for item in re.split(r"[\s,]+", value.strip()):
        if not item:
            continue
        try:
            user_ids.add(int(item))
        except ValueError as exc:
            raise TelegramConfigError(f"Invalid Telegram user id: {item}") from exc
    return frozenset(user_ids)


def is_authorized(user_id: int | None, allowed_user_ids: frozenset[int]) -> bool:
    return user_id is not None and user_id in allowed_user_ids


def parse_run_command(text: str) -> RunCommand:
    parts = text.strip().split(maxsplit=2)
    if not parts:
        raise TelegramCommandError("Use /run <goal>.")

    command = parts[0].split("@", maxsplit=1)[0].lower()
    if command != "/run":
        raise TelegramCommandError("Use /run <goal>.")
    if len(parts) < 2:
        raise TelegramCommandError("Use /run <goal> or /run general <goal>.")

    requested_mode = parts[1].lower()
    if requested_mode in {"general", "ml"}:
        if len(parts) < 3 or not parts[2].strip():
            raise TelegramCommandError(f"Use /run {requested_mode} <goal>.")
        return RunCommand(mode=requested_mode, goal=parts[2].strip())  # type: ignore[arg-type]

    goal = text.strip()[len(parts[0]) :].strip()
    if not goal:
        raise TelegramCommandError("Use /run <goal>.")
    return RunCommand(mode="general", goal=goal)


def load_telegram_settings() -> TelegramSettings:
    load_dotenv(Path.cwd() / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise TelegramConfigError("TELEGRAM_BOT_TOKEN is required.")

    allowed_user_ids = parse_allowed_user_ids(os.getenv("TELEGRAM_ALLOWED_USER_IDS", ""))
    if not allowed_user_ids:
        raise TelegramConfigError("TELEGRAM_ALLOWED_USER_IDS must contain at least one Telegram user id.")

    return TelegramSettings(
        bot_token=token,
        allowed_user_ids=allowed_user_ids,
        workspace_dir=os.getenv("EVORESEARCHER_TELEGRAM_WORKSPACE_DIR") or None,
        search_enabled=os.getenv("EVORESEARCHER_TELEGRAM_NO_SEARCH", "").strip().lower()
        not in {"1", "true", "yes"},
        tree_depth=int(os.getenv("EVORESEARCHER_TELEGRAM_TREE_DEPTH", "2")),
        branching_factor=int(os.getenv("EVORESEARCHER_TELEGRAM_BRANCHING_FACTOR", "2")),
        max_sources=int(os.getenv("EVORESEARCHER_TELEGRAM_MAX_SOURCES", "6")),
    )


class TelegramObserver:
    """Observer that forwards useful progress and intermediate results to Telegram."""

    def __init__(
        self,
        *,
        bot: Any,
        chat_id: int,
        loop: asyncio.AbstractEventLoop,
        status_callback,
        selection_callback,
        min_log_interval_seconds: float = 12.0,
    ) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.loop = loop
        self.status_callback = status_callback
        self.selection_callback = selection_callback
        self.min_log_interval_seconds = min_log_interval_seconds
        self._last_log_at = 0.0
        self._last_phase = ""
        self._lock = threading.Lock()

    def start_run(
        self,
        *,
        run_id: str,
        mode: str,
        goal: str,
        model_name: str,
        provider: str,
        workspace_dir: Path,
    ) -> None:
        self._set_status(f"started {run_id} ({mode})")
        self._send(
            "Great. I have started the research run.\n\n"
            f"Task type: {mode}\n"
            f"Model: {model_name}\n"
            f"Goal: {goal}"
        )

    def set_phase(self, phase: str, status: str) -> None:
        with self._lock:
            phase_changed = phase != self._last_phase
            self._last_phase = phase
        self._set_status(f"{phase}: {status}")
        if phase_changed:
            self._send(_natural_phase_message(phase, status))

    def phase_log(self, phase: str, message: str) -> None:
        self._set_status(f"{phase}: {message}")
        now = time.monotonic()
        with self._lock:
            if now - self._last_log_at < self.min_log_interval_seconds:
                return
            self._last_log_at = now
        self._send(_natural_phase_message(phase, message))

    def metric(self, name: str, value) -> None:
        self._set_status(f"{name}: {value}")

    def agent_state(self, name: str, status: str, detail: str) -> None:
        self._set_status(f"{name}: {status} - {detail}")

    def artifact(self, label: str, path: Path) -> None:
        self._set_status(f"artifact ready: {label}")

    def memories_ready(
        self,
        *,
        memory_hits: list[MemoryEntry],
        proposal_hits: list[MemoryEntry],
        ideation_backend: str,
        proposal_backend: str,
    ) -> None:
        self._send(
            _format_memory_preview(
                memory_hits=memory_hits,
                proposal_hits=proposal_hits,
                ideation_backend=ideation_backend,
                proposal_backend=proposal_backend,
            )
        )

    def candidate_ideas_ready(self, ideas: list[ResearchIdea], *, stage: str) -> None:
        self._send(_format_candidate_preview(ideas, stage=stage))

    def ideas_ready(self, ideas: list[ResearchIdea]) -> None:
        self._send(_format_ideas_preview(ideas))

    def report_ready(self, report: ReportSections) -> None:
        abstract = _trim(report.abstract, 900)
        self._send(
            "The final report draft is now assembled. Here is the opening summary before I render files:\n\n"
            f"{report.title}\n\n"
            f"{abstract}"
        )

    def finish(self, message: str) -> None:
        self._set_status(message)
        self._send("The run is complete. I am sending the report files now.")

    def select_option(
        self,
        *,
        title: str,
        prompt: str,
        options: list[dict],
        custom_prompt: str,
        question_index: int,
        total_questions: int,
        selected_answers: list[str],
    ) -> str:
        return self.selection_callback(
            title=title,
            prompt=prompt,
            options=options,
            custom_prompt=custom_prompt,
            question_index=question_index,
            total_questions=total_questions,
            selected_answers=selected_answers,
        )

    def _set_status(self, status: str) -> None:
        self.status_callback(status)

    def _send(self, text: str) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self.bot.send_message(chat_id=self.chat_id, text=text[:3900]),
            self.loop,
        )
        future.add_done_callback(_discard_send_result)


class TelegramResearchBot:
    def __init__(self, settings: TelegramSettings) -> None:
        self.settings = settings
        self.active_tasks: dict[int, asyncio.Task] = {}
        self.last_runs: dict[int, RunResult] = {}
        self.status_by_chat: dict[int, str] = {}
        self.pending_mode_by_chat: dict[int, ModeName] = {}
        self.pending_model_by_chat: dict[int, ModelPreset] = {}
        self.selection_waiters: dict[int, PendingSelection] = {}
        self.custom_selection_by_chat: dict[int, int] = {}
        self._selection_ids = count(1)
        self._lock = threading.Lock()

    async def start(self, update, context) -> None:
        if not await self._ensure_authorized(update):
            return
        chat_id = update.effective_chat.id
        if chat_id in self.active_tasks:
            await update.effective_message.reply_text(
                "I am already working on a run for this chat. Use /status for the latest checkpoint."
            )
            return
        await self._ask_for_mode(update.effective_message)

    async def help(self, update, context) -> None:
        if not await self._ensure_authorized(update):
            return
        await update.effective_message.reply_text(COMMANDS)

    async def text(self, update, context) -> None:
        if not await self._ensure_authorized(update):
            return
        chat_id = update.effective_chat.id
        text = (update.effective_message.text or "").strip()
        if not text:
            return

        if await self._maybe_accept_custom_selection(update, text):
            return

        if chat_id in self.active_tasks:
            await update.effective_message.reply_text(
                "I am still working on the current run. Use /status if you want the latest checkpoint."
            )
            return

        mode = self.pending_mode_by_chat.get(chat_id)
        if mode is None:
            await update.effective_message.reply_text(
                "Let's choose the task type first."
            )
            await self._ask_for_mode(update.effective_message)
            return
        model_preset = self.pending_model_by_chat.get(chat_id)
        if model_preset is None:
            await update.effective_message.reply_text("Choose the model profile before sending the goal.")
            await self._ask_for_model(update.effective_message)
            return

        self.pending_mode_by_chat.pop(chat_id, None)
        self.pending_model_by_chat.pop(chat_id, None)
        await self._start_run(
            chat_id=chat_id,
            mode=mode,
            goal=text,
            bot=context.bot,
            model_preset=model_preset,
        )

    async def status(self, update, context) -> None:
        if not await self._ensure_authorized(update):
            return
        chat_id = update.effective_chat.id
        if chat_id in self.active_tasks:
            await update.effective_message.reply_text(
                "Current checkpoint: " + self.status_by_chat.get(chat_id, "the run is active.")
            )
            return
        if chat_id in self.pending_mode_by_chat:
            if chat_id not in self.pending_model_by_chat:
                await update.effective_message.reply_text("I am waiting for your model profile choice.")
            else:
                await update.effective_message.reply_text("I am waiting for your research goal.")
            return
        await update.effective_message.reply_text("No active run. Use /start to begin.")

    async def last(self, update, context) -> None:
        if not await self._ensure_authorized(update):
            return
        chat_id = update.effective_chat.id
        result = self.last_runs.get(chat_id)
        if result is None:
            await update.effective_message.reply_text("No completed run for this chat yet.")
            return
        await update.effective_message.reply_text(_format_done_message(result))
        await self._send_artifacts(context.bot, chat_id, result)

    async def cancel(self, update, context) -> None:
        if not await self._ensure_authorized(update):
            return
        chat_id = update.effective_chat.id
        if chat_id in self.custom_selection_by_chat:
            self.custom_selection_by_chat.pop(chat_id, None)
            await update.effective_message.reply_text("Cancelled the custom answer. Please tap an option instead.")
            return
        if chat_id not in self.active_tasks:
            self.pending_mode_by_chat.pop(chat_id, None)
            self.pending_model_by_chat.pop(chat_id, None)
            await update.effective_message.reply_text("No active run. Use /start to begin again.")
            return
        await update.effective_message.reply_text(
            "The research pipeline is already executing and cannot be interrupted safely yet. "
            "I will keep sending progress here."
        )

    async def unknown(self, update, context) -> None:
        if not await self._ensure_authorized(update):
            return
        await update.effective_message.reply_text(
            "I do not need commands for goals. Use /start, choose a type and model, then type your goal."
        )

    async def handle_callback(self, update, context) -> None:
        if not await self._ensure_authorized(update):
            return
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        data = query.data or ""
        if data.startswith(MODE_CALLBACK_PREFIX):
            await self._handle_mode_callback(query, data)
            return
        if data.startswith(MODEL_CALLBACK_PREFIX):
            await self._handle_model_callback(query, data)
            return
        if data.startswith(ML_CUSTOM_CALLBACK_PREFIX):
            await self._handle_ml_custom_callback(query, data)
            return
        if data.startswith(ML_CALLBACK_PREFIX):
            await self._handle_ml_option_callback(query, data)
            return

    def request_selection(
        self,
        *,
        chat_id: int,
        bot,
        loop: asyncio.AbstractEventLoop,
        title: str,
        prompt: str,
        options: list[dict],
        custom_prompt: str,
        question_index: int,
        total_questions: int,
        selected_answers: list[str],
    ) -> str:
        if not options:
            return ""
        request_id = next(self._selection_ids)
        future: Future[str] = Future()
        pending = PendingSelection(
            chat_id=chat_id,
            request_id=request_id,
            title=title,
            options=options,
            future=future,
        )
        with self._lock:
            self.selection_waiters[request_id] = pending
        send_future = asyncio.run_coroutine_threadsafe(
            self._send_ml_question(
                bot=bot,
                chat_id=chat_id,
                request_id=request_id,
                title=title,
                prompt=prompt,
                options=options,
                custom_prompt=custom_prompt,
                question_index=question_index,
                total_questions=total_questions,
                selected_answers=selected_answers,
            ),
            loop,
        )
        send_future.result(timeout=30)
        try:
            answer = future.result(timeout=3600)
        finally:
            with self._lock:
                self.selection_waiters.pop(request_id, None)
                if self.custom_selection_by_chat.get(chat_id) == request_id:
                    self.custom_selection_by_chat.pop(chat_id, None)
        return answer

    async def _ask_for_mode(self, message) -> None:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("General research", callback_data=f"{MODE_CALLBACK_PREFIX}general"),
                    InlineKeyboardButton("ML research", callback_data=f"{MODE_CALLBACK_PREFIX}ml"),
                ]
            ]
        )
        await message.reply_text("Which kind of task do you want to work on?", reply_markup=keyboard)

    async def _handle_mode_callback(self, query, data: str) -> None:
        chat_id = query.message.chat_id
        mode = data[len(MODE_CALLBACK_PREFIX) :]
        if mode not in {"general", "ml"}:
            await query.edit_message_text("That task type is not supported. Use /start to choose again.")
            return
        if chat_id in self.active_tasks:
            await query.edit_message_text("I am already working on a run for this chat. Use /status for progress.")
            return
        self.pending_mode_by_chat[chat_id] = mode
        self.pending_model_by_chat.pop(chat_id, None)
        await query.edit_message_text(
            f"Great. We will work in {mode} mode. Now choose the model profile.",
            reply_markup=self._model_keyboard(),
        )

    async def _ask_for_model(self, message) -> None:
        await message.reply_text("Which model profile should I use?", reply_markup=self._model_keyboard())

    def _model_keyboard(self):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(preset.label, callback_data=f"{MODEL_CALLBACK_PREFIX}{preset.preset_id}")]
                for preset in MODEL_PRESETS
            ]
        )

    async def _handle_model_callback(self, query, data: str) -> None:
        chat_id = query.message.chat_id
        if chat_id in self.active_tasks:
            await query.edit_message_text("I am already working on a run for this chat. Use /status for progress.")
            return
        if chat_id not in self.pending_mode_by_chat:
            await query.edit_message_text("Choose the task type first. Use /start to begin again.")
            return
        preset_id = data[len(MODEL_CALLBACK_PREFIX) :]
        preset = resolve_model_preset(preset_id)
        if preset is None:
            await query.edit_message_text("That model profile is not supported. Use /start to choose again.")
            return
        self.pending_model_by_chat[chat_id] = preset
        await query.edit_message_text(
            f"Great. I will use {preset.label}.\n\n{preset.description}\n\nHow can I help you?"
        )

    async def _start_run(
        self,
        *,
        chat_id: int,
        mode: ModeName,
        goal: str,
        bot,
        model_preset: ModelPreset,
    ) -> None:
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "Got it. I will turn that into a research brief and report back as the pipeline makes progress.\n\n"
                f"Model profile: {model_preset.label}"
            ),
        )
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(
            self._run_job(
                chat_id,
                RunCommand(
                    mode=mode,
                    goal=goal,
                    deepseek_model=model_preset.deepseek_model,
                    deepseek_reasoning_model=model_preset.deepseek_reasoning_model,
                    model_label=model_preset.label,
                ),
                bot,
                loop,
            )
        )
        self.active_tasks[chat_id] = task
        task.add_done_callback(lambda finished: self.active_tasks.pop(chat_id, None))

    async def _run_job(self, chat_id: int, command: RunCommand, bot, loop: asyncio.AbstractEventLoop) -> None:
        observer = TelegramObserver(
            bot=bot,
            chat_id=chat_id,
            loop=loop,
            status_callback=lambda status: self._set_status(chat_id, status),
            selection_callback=lambda **kwargs: self.request_selection(
                chat_id=chat_id,
                bot=bot,
                loop=loop,
                **kwargs,
            ),
        )
        options = RunOptions(
            workspace_dir=self.settings.workspace_dir,
            search_enabled=self.settings.search_enabled,
            tree_depth=self.settings.tree_depth,
            branching_factor=self.settings.branching_factor,
            max_sources=self.settings.max_sources,
            deepseek_model=command.deepseek_model,
            deepseek_reasoning_model=command.deepseek_reasoning_model,
        )
        try:
            result = await asyncio.to_thread(
                run_research,
                goal=command.goal,
                mode=command.mode,
                options=options,
                observer=observer,
            )
        except Exception as exc:
            self._set_status(chat_id, f"failed: {exc}")
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "The run failed before the final files were ready.\n\n"
                    f"Error: {exc}\n\n"
                    "You can use /start to try a new run."
                ),
            )
            return

        self.last_runs[chat_id] = result
        self._set_status(chat_id, "completed")
        await bot.send_message(chat_id=chat_id, text=_format_done_message(result))
        await self._send_artifacts(bot, chat_id, result)

    async def _send_ml_question(
        self,
        *,
        bot,
        chat_id: int,
        request_id: int,
        title: str,
        prompt: str,
        options: list[dict],
        custom_prompt: str,
        question_index: int,
        total_questions: int,
        selected_answers: list[str],
    ) -> None:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        rows = []
        for idx, option in enumerate(options):
            label = str(option.get("label") or option.get("value") or f"Option {idx + 1}")
            rows.append([InlineKeyboardButton(label[:48], callback_data=f"{ML_CALLBACK_PREFIX}{request_id}:{idx}")])
        rows.append([InlineKeyboardButton("Custom answer", callback_data=f"{ML_CUSTOM_CALLBACK_PREFIX}{request_id}")])
        await bot.send_message(
            chat_id=chat_id,
            text=_format_selection_question_text(
                prompt=prompt,
                question_index=question_index,
                total_questions=total_questions,
                selected_answers=selected_answers,
            ),
            reply_markup=InlineKeyboardMarkup(rows),
        )

    async def _handle_ml_option_callback(self, query, data: str) -> None:
        parsed = data[len(ML_CALLBACK_PREFIX) :].split(":", maxsplit=1)
        if len(parsed) != 2:
            await query.edit_message_text("That option is no longer valid.")
            return
        try:
            request_id = int(parsed[0])
            option_index = int(parsed[1])
        except ValueError:
            await query.edit_message_text("That option is no longer valid.")
            return
        pending = self.selection_waiters.get(request_id)
        if pending is None or pending.chat_id != query.message.chat_id:
            await query.edit_message_text("That question has expired.")
            return
        if option_index >= len(pending.options):
            await query.edit_message_text("That option is no longer valid.")
            return
        option = pending.options[option_index]
        value = str(option.get("value", ""))
        label = str(option.get("label", value))
        if not pending.future.done():
            pending.future.set_result(value)
        await query.edit_message_text(f"Selected: {label}")

    async def _handle_ml_custom_callback(self, query, data: str) -> None:
        try:
            request_id = int(data[len(ML_CUSTOM_CALLBACK_PREFIX) :])
        except ValueError:
            await query.edit_message_text("That question is no longer valid.")
            return
        pending = self.selection_waiters.get(request_id)
        if pending is None or pending.chat_id != query.message.chat_id:
            await query.edit_message_text("That question has expired.")
            return
        self.custom_selection_by_chat[pending.chat_id] = request_id
        await query.edit_message_text("Custom answer")
        await query.message.reply_text("Type your custom answer in the next message.")

    async def _maybe_accept_custom_selection(self, update, text: str) -> bool:
        chat_id = update.effective_chat.id
        request_id = self.custom_selection_by_chat.get(chat_id)
        if request_id is None:
            return False
        pending = self.selection_waiters.get(request_id)
        if pending is None:
            self.custom_selection_by_chat.pop(chat_id, None)
            await update.effective_message.reply_text("That question has expired. Please wait for the next prompt.")
            return True
        if not pending.future.done():
            pending.future.set_result(text)
        self.custom_selection_by_chat.pop(chat_id, None)
        await update.effective_message.reply_text("Got it. I will use that answer.")
        return True

    async def _send_artifacts(self, bot, chat_id: int, result: RunResult) -> None:
        for filename in ["research_report.pdf", "research_report.md", "run_summary.json", "pdf_render_warning.txt"]:
            path = result.run_dir / filename
            if not path.exists():
                continue
            with path.open("rb") as file_handle:
                await bot.send_document(chat_id=chat_id, document=file_handle, filename=path.name)

    async def _ensure_authorized(self, update) -> bool:
        user_id = update.effective_user.id if update.effective_user is not None else None
        if is_authorized(user_id, self.settings.allowed_user_ids):
            return True
        message = "This Telegram user is not allowed."
        if user_id is not None:
            message += f"\nYour Telegram user id is: {user_id}"
        target = update.callback_query.message if update.callback_query is not None else update.effective_message
        await target.reply_text(message)
        return False

    def _set_status(self, chat_id: int, status: str) -> None:
        with self._lock:
            self.status_by_chat[chat_id] = status


def _natural_phase_message(phase: str, detail: str) -> str:
    if phase == "intake":
        return (
            "I am turning your message into a precise research brief. "
            "Next I will retrieve relevant past memories, gather evidence, then grow and rank candidate ideas."
        )
    if phase == "research" and detail.startswith("Searching web for:"):
        return "I am checking external evidence now.\n\n" + detail
    if phase == "research" and "Expanded tree depth" in detail:
        return (
            "I expanded another layer of the idea tree. "
            "Next I will score these directions, keep the strongest branches, and move toward head-to-head ranking."
        )
    if phase == "research":
        return (
            "I am gathering context and building candidate ideas. "
            "Right now I am using prior run memories as constraints and inspiration, then I will search evidence "
            "and draft the first candidate direction.\n\n"
            + detail
        )
    if phase == "ranking":
        return (
            "The candidate ideas are ready. I am ranking them head-to-head now so the final report uses "
            "the strongest, most feasible direction rather than the first plausible one."
        )
    if phase == "proposal":
        return (
            "I have selected the strongest directions and am writing the report. "
            "This is the stage that turns ranked ideas and evidence into a structured proposal."
        )
    if phase == "publish":
        return (
            "The report text is ready. I am writing Markdown and LaTeX now, then I will try PDF rendering. "
            "If PDF rendering fails, I will still send the Markdown report and the render warning."
        )
    if phase == "memory":
        return "I am saving what this run learned into EvoResearcher's memory."
    return detail


def _format_selection_question_text(
    *,
    prompt: str,
    question_index: int,
    total_questions: int,
    selected_answers: list[str],
) -> str:
    context = ""
    if selected_answers:
        context = "\n\nAlready selected:\n" + "\n".join(f"- {answer}" for answer in selected_answers[-3:])
    return (
        f"Question {question_index}/{total_questions}\n\n"
        f"{prompt}"
        f"{context}\n\n"
        f"Tap an option, or choose custom to type your own answer."
    )[:3900]


def _format_memory_preview(
    *,
    memory_hits: list[MemoryEntry],
    proposal_hits: list[MemoryEntry],
    ideation_backend: str,
    proposal_backend: str,
) -> str:
    lines = [
        "I found prior memory that may shape this run.",
        "",
        f"Ideation memories: {len(memory_hits)} via {ideation_backend}",
        f"Proposal memories: {len(proposal_hits)} via {proposal_backend}",
    ]
    samples = [*memory_hits[:1], *proposal_hits[:1]]
    if samples:
        lines.append("")
        lines.append("Example memory I will reuse:")
        for entry in samples:
            lines.append(
                "\n"
                f"- {entry.summary}\n"
                f"  {_trim(entry.details, 420)}"
            )
    lines.append(
        "\nNext I will use these memories to avoid repeating weak directions and to seed better candidate ideas."
    )
    return "\n".join(lines)[:3900]


def _format_candidate_preview(ideas: list[ResearchIdea], *, stage: str) -> str:
    if not ideas:
        return f"{stage}: no candidate ideas were produced at this checkpoint."
    lines = [f"Candidate checkpoint: {stage}", ""]
    for idx, idea in enumerate(ideas[:2], start=1):
        score = f"{idea.total_score:.2f}" if idea.total_score else "pending"
        lines.append(
            f"{idx}. {idea.title}\n"
            f"Score: {score}\n"
            f"Why it matters: {_trim(idea.summary, 360)}\n"
            f"Method sketch: {_trim(idea.method_outline, 360)}"
        )
        if idx < min(len(ideas), 2):
            lines.append("")
    lines.append("\nI will keep expanding and comparing these before writing the final report.")
    return "\n".join(lines)[:3900]


def _format_ideas_preview(ideas: list[ResearchIdea]) -> str:
    if not ideas:
        return "The idea generation stage finished, but no ranked ideas were returned."
    lines = ["I have the first ranked idea set. Here are the strongest directions so far:"]
    for idx, idea in enumerate(ideas[:3], start=1):
        score = f"{idea.total_score:.2f}" if idea.total_score else "n/a"
        lines.append(
            "\n"
            f"{idx}. {idea.title}\n"
            f"Score: {score}\n"
            f"{_trim(idea.summary, 520)}"
        )
    lines.append("\nI will now synthesize this into the final report.")
    return "\n".join(lines)[:3900]


def _format_done_message(result: RunResult) -> str:
    artifacts = result.state.get("artifacts", {})
    artifact_lines = "\n".join(f"- {label}: {path}" for label, path in artifacts.items())
    if artifact_lines:
        artifact_lines = "\n\nFiles generated:\n" + artifact_lines
    pdf_note = ""
    if "pdf_warning_path" in artifacts and "pdf_path" not in artifacts:
        pdf_note = (
            "\n\nPDF rendering did not complete, so I am sending the Markdown report plus the render warning. "
            "The research output itself is still available."
        )
    return (
        "Done. The report files are attached below.\n\n"
        f"Run id: {result.run_id}\n"
        f"Directory: {result.run_dir}"
        f"{artifact_lines}"
        f"{pdf_note}"
    )


def _trim(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _discard_send_result(future) -> None:
    try:
        future.exception()
    except (asyncio.CancelledError, FutureCancelledError):
        pass


def build_application(settings: TelegramSettings):
    try:
        from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters
    except ImportError as exc:
        raise TelegramConfigError(
            "python-telegram-bot is not installed. Install with: pip install '.[telegram]'"
        ) from exc

    bot = TelegramResearchBot(settings)
    application = Application.builder().token(settings.bot_token).build()
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help))
    application.add_handler(CommandHandler("status", bot.status))
    application.add_handler(CommandHandler("last", bot.last))
    application.add_handler(CommandHandler("cancel", bot.cancel))
    application.add_handler(CallbackQueryHandler(bot.handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.text))
    application.add_handler(MessageHandler(filters.COMMAND, bot.unknown))
    return application


def main() -> None:
    settings = load_telegram_settings()
    application = build_application(settings)
    print("EvoResearcher Telegram bot is running. Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()
