"""Runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import re

from dotenv import dotenv_values


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned[:48] or "research-run"


@dataclass(slots=True)
class AppConfig:
    workspace_dir: Path
    outputs_dir: Path
    memory_dir: Path
    author_line: str
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str
    deepseek_reasoning_model: str = "deepseek-v4-pro"
    max_sources: int = 6
    tree_depth: int = 2
    branching_factor: int = 2
    max_report_pages: int = 3
    search_enabled: bool = True

    def ensure_directories(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def make_run_id(self, goal: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return f"{stamp}-{_slugify(goal)}"


def load_config(
    *,
    workspace_dir: str | None = None,
    search_enabled: bool = True,
    tree_depth: int | None = None,
    branching_factor: int | None = None,
    max_sources: int | None = None,
    deepseek_model: str | None = None,
    deepseek_reasoning_model: str | None = None,
) -> AppConfig:
    root = Path(workspace_dir or Path.cwd()).resolve()
    env_values = dotenv_values(root / ".env")

    def env_value(name: str, default: str = "") -> str:
        process_value = os.getenv(name)
        if process_value:
            return process_value
        file_value = env_values.get(name)
        if file_value is None:
            return default
        return str(file_value)

    outputs_dir = root / "outputs"
    memory_dir = root / "memory"
    deepseek_api_key = env_value("DEEPSEEK_API_KEY")
    resolved_deepseek_model = deepseek_model or env_value("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_base_url = env_value(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com/chat/completions",
    )
    resolved_deepseek_reasoning_model = (
        deepseek_reasoning_model or env_value("DEEPSEEK_REASONING_MODEL", "deepseek-v4-pro")
    )
    author_line = env_value("EVORESEARCHER_AUTHOR", "EvoResearcher")
    if not deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required.")
    config = AppConfig(
        workspace_dir=root,
        outputs_dir=outputs_dir,
        memory_dir=memory_dir,
        author_line=author_line,
        deepseek_api_key=deepseek_api_key,
        deepseek_model=resolved_deepseek_model,
        deepseek_base_url=deepseek_base_url,
        deepseek_reasoning_model=resolved_deepseek_reasoning_model,
        max_sources=max_sources or 6,
        tree_depth=tree_depth or 2,
        branching_factor=branching_factor or 2,
        search_enabled=search_enabled,
    )
    config.ensure_directories()
    return config
