"""Rich live observer with a multi-panel EvoScientist-like dashboard."""

from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
import os
import threading
import time

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from prompt_toolkit.shortcuts import button_dialog, input_dialog
from prompt_toolkit.styles import Style


_PHASE_ORDER = ["bootstrap", "intake", "research", "ranking", "proposal", "publish", "memory"]
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_INTAKE_STYLE = Style.from_dict(
    {
        "dialog": "bg:#0f172a",
        "dialog frame.label": "bg:#1d4ed8 #eff6ff bold",
        "dialog.body": "bg:#0b1220 #dbeafe",
        "dialog shadow": "bg:#020617",
        "button": "bg:#1e293b #93c5fd",
        "button.focused": "bg:#38bdf8 #082f49 bold",
        "text-area": "bg:#0f172a #e2e8f0",
    }
)
_TITLE_TEXT = "EVORESEARCHER"


@dataclass
class UIState:
    run_id: str = ""
    goal: str = ""
    mode: str = ""
    model_name: str = ""
    provider: str = ""
    workspace_dir: str = ""
    ui_name: str = "tui"
    started_at: float = 0.0
    phase: str = "idle"
    status: str = "ready"
    phase_status: dict[str, str] = field(
        default_factory=lambda: {phase: "pending" for phase in _PHASE_ORDER}
    )
    metrics: dict[str, str] = field(default_factory=dict)
    agents: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "intake": ("idle", "waiting"),
            "research": ("idle", "waiting"),
            "proposal": ("idle", "waiting"),
            "ema": ("idle", "waiting"),
            "publish": ("idle", "waiting"),
        }
    )
    artifacts: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=8))
    events: deque[str] = field(default_factory=lambda: deque(maxlen=14))


class RichObserver(AbstractContextManager):
    def __init__(self) -> None:
        self.console = Console()
        self.state = UIState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._live: Live | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._live = Live(self._render(), console=self.console, screen=True, refresh_per_second=10)
        self._live.start()
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._live is not None:
            self._live.update(self._render(final=True))
            self._live.stop()

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
        with self._lock:
            self.state.run_id = run_id
            self.state.mode = mode
            self.state.goal = goal
            self.state.model_name = model_name
            self.state.provider = provider
            self.state.workspace_dir = str(workspace_dir)
            self.state.started_at = time.monotonic()
            self.state.phase = "bootstrap"
            self.state.phase_status["bootstrap"] = "active"
            self.state.status = "initializing agent graph"
            self.state.events.appendleft(f"Run {run_id} queued")

    def set_phase(self, phase: str, status: str) -> None:
        with self._lock:
            previous = self.state.phase
            if previous in self.state.phase_status and self.state.phase_status[previous] == "active":
                self.state.phase_status[previous] = "done"
            self.state.phase = phase
            if phase in self.state.phase_status:
                self.state.phase_status[phase] = "active"
            self.state.status = status
            self.state.events.appendleft(f"{phase}: {status}")

    def phase_log(self, phase: str, message: str) -> None:
        with self._lock:
            self.state.phase = phase
            self.state.status = message
            self.state.events.appendleft(message)

    def metric(self, name: str, value) -> None:
        with self._lock:
            self.state.metrics[name] = str(value)

    def agent_state(self, name: str, status: str, detail: str) -> None:
        with self._lock:
            self.state.agents[name] = (status, detail)
            self.state.events.appendleft(f"agent {name}: {status} :: {detail}")

    def artifact(self, label: str, path: Path) -> None:
        with self._lock:
            self.state.artifacts.appendleft((label, str(path)))
            self.state.events.appendleft(f"artifact ready: {label}")

    def finish(self, message: str) -> None:
        with self._lock:
            self.state.status = message
            if self.state.phase in self.state.phase_status:
                self.state.phase_status[self.state.phase] = "done"
            self.state.events.appendleft(message)

    def prompt_user(self, title: str, prompt: str, *, default: str = "") -> str:
        with self._lock:
            self.state.status = f"Waiting for input: {title}"
            self.state.events.appendleft(f"input requested: {title}")
        self._paused.set()
        if self._live is not None:
            self._live.stop()
        try:
            panel = Panel(prompt, title=title, border_style="cyan")
            self.console.print(panel)
            response = self.console.input("[bold cyan]> [/bold cyan]")
            return response if response.strip() else default
        finally:
            if self._live is not None:
                self._live.start()
            self._paused.clear()

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
        with self._lock:
            self.state.status = f"Waiting for selection: {title}"
            self.state.events.appendleft(f"selection requested: {title}")
        self._paused.set()
        if self._live is not None:
            self._live.stop()
        try:
            progress_line = f"Question {question_index}/{total_questions}"
            chosen = "\n".join([f"- {answer}" for answer in selected_answers]) or "- None yet"
            rendered_prompt = (
                f"{progress_line}\n\n"
                f"{prompt}\n\n"
                "Use Left/Right to move, Enter to confirm.\n\n"
                "Options:\n"
                + "\n".join(
                    [f"- {option['label']}: {option.get('description', '')}" for option in options]
                )
                + "\n\nSelected so far:\n"
                + chosen
            )
            buttons = [(option["label"], option["value"]) for option in options]
            buttons.append(("Custom", "__custom__"))
            selection = button_dialog(
                title=title,
                text=rendered_prompt,
                buttons=buttons,
                style=_INTAKE_STYLE,
            ).run()
            if selection == "__custom__":
                custom_value = input_dialog(
                    title=title,
                    text=custom_prompt,
                    style=_INTAKE_STYLE,
                ).run()
                return "" if custom_value is None else custom_value.strip()
            return "" if selection is None else str(selection).strip()
        finally:
            if self._live is not None:
                self._live.start()
            self._paused.clear()

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.1)
                continue
            if self._live is not None:
                self._live.update(self._render())
            time.sleep(0.1)

    def _render(self, final: bool = False):
        elapsed = 0.0
        if self.state.started_at:
            elapsed = time.monotonic() - self.state.started_at
        spinner = _SPINNER[int(elapsed * 10) % len(_SPINNER)] if not final else "●"
        banner = Text()
        banner.append(_TITLE_TEXT + "\n", style="bold #60a5fa")
        meta = [
            ("Model", self.state.model_name or "n/a", "#67e8f9"),
            ("Provider", self.state.provider or "n/a", "#f472b6"),
            ("Mode", self.state.mode or "n/a", "#f59e0b"),
            ("UI", self.state.ui_name, "#a78bfa"),
        ]
        for idx, (label, value, color) in enumerate(meta):
            banner.append(f"{label}: ", style="bold #cbd5e1")
            banner.append(value, style=f"bold {color}")
            if idx < len(meta) - 1:
                banner.append("   ", style="bold #475569")
        banner.append("\n", style="bold #475569")
        banner.append("Directory: ", style="bold #cbd5e1")
        banner.append(self.state.workspace_dir or os.getcwd(), style="bold #fda4af")
        banner.append("\n", style="bold #475569")
        banner.append("Type / for commands", style="bold #fde68a")
        header = Panel(banner, border_style="#2563eb")
        mission = Panel(
            f"[bold white]Goal[/bold white]\n{self.state.goal or 'Waiting for input...'}\n\n"
            f"[bold white]Status[/bold white]\n{self.state.status}",
            title="Mission Control",
            border_style="#38bdf8",
        )
        phase_table = Table(show_header=False, box=None, padding=(0, 1))
        completed = 0
        for phase in _PHASE_ORDER:
            state = self.state.phase_status.get(phase, "pending")
            icon = "●"
            style = "dim"
            if state == "active":
                icon = spinner
                style = "bold yellow"
            elif state == "done":
                icon = "■"
                style = "bold green"
                completed += 1
            phase_table.add_row(Text(icon, style=style), Text(phase, style=style))
        progress = ProgressBar(total=len(_PHASE_ORDER), completed=completed, width=20)
        phase_panel = Panel(
            Group(phase_table, Align.left(progress)),
            title="Pipeline",
            border_style="#f59e0b",
        )
        agent_table = Table(show_header=True, header_style="bold white", box=None, padding=(0, 1))
        agent_table.add_column("Agent")
        agent_table.add_column("State")
        agent_table.add_column("Detail")
        for name, (status, detail) in self.state.agents.items():
            style = {
                "active": "bold yellow",
                "done": "bold green",
                "idle": "dim",
            }.get(status, "white")
            agent_table.add_row(name, Text(status, style=style), detail)
        agent_panel = Panel(agent_table, title="Agents", border_style="#f472b6")
        metrics_table = Table(show_header=False, box=None, padding=(0, 1))
        for key in ["memory_hits", "sources", "tree_nodes", "leaf_nodes", "elo_matches", "top_elo"]:
            metrics_table.add_row(key, self.state.metrics.get(key, "0"))
        metrics_panel = Panel(metrics_table, title="Metrics", border_style="#22c55e")
        events_table = Table.grid(padding=(0, 1))
        for event in list(self.state.events)[:10]:
            events_table.add_row(f"• {event}")
        events_panel = Panel(events_table, title="Live Events", border_style="#a78bfa")
        artifacts_table = Table.grid(padding=(0, 1))
        for label, path in list(self.state.artifacts)[:6]:
            artifacts_table.add_row(f"[green]{label}[/green]", path)
        artifacts_panel = Panel(artifacts_table, title="Artifacts", border_style="#14b8a6")
        layout = Layout()
        layout.split_column(
            Layout(header, size=12),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2),
        )
        layout["left"].split_column(
            Layout(mission, ratio=3),
            Layout(events_panel, ratio=4),
        )
        layout["right"].split_column(
            Layout(phase_panel, ratio=3),
            Layout(agent_panel, ratio=3),
            Layout(Columns([metrics_panel, artifacts_panel], equal=True), ratio=4),
        )
        return layout
