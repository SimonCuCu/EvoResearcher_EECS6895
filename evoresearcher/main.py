"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json

from evoresearcher.runner import RunOptions, run_research
from evoresearcher.tui.observer import RichObserver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EvoResearcher.")
    parser.add_argument("--goal", default=None, help="Research goal or benchmark-style query.")
    parser.add_argument("--mode", choices=("general", "ml"), default="general")
    parser.add_argument("--workspace-dir", default=None)
    parser.add_argument("--tree-depth", type=int, default=2, help="Idea tree search depth.")
    parser.add_argument("--branching-factor", type=int, default=2, help="Number of children kept per expansion step.")
    parser.add_argument("--max-sources", type=int, default=6, help="Maximum number of retrieved web sources.")
    parser.add_argument("--model", default=None, help="Override the default model used by intake/proposal/EMA calls.")
    parser.add_argument("--reasoning-model", default=None, help="Override the model used by research reasoning calls.")
    parser.add_argument(
        "--global-model",
        default=None,
        help="Override both default and research reasoning models, e.g. deepseek-v4-flash.",
    )
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    goal = args.goal or input("Enter your research question: ").strip()
    if not goal:
        raise SystemExit("A non-empty goal is required.")
    options = RunOptions(
        workspace_dir=args.workspace_dir,
        search_enabled=not args.no_search,
        tree_depth=args.tree_depth,
        branching_factor=args.branching_factor,
        max_sources=args.max_sources,
        deepseek_model=args.global_model or args.model,
        deepseek_reasoning_model=args.global_model or args.reasoning_model,
    )
    with RichObserver() as observer:
        result = run_research(
            goal=goal,
            mode=args.mode,
            options=options,
            observer=observer,
        )
    if args.print_json:
        print(json.dumps(result.state, indent=2))
    else:
        print(f"Run completed: {result.run_dir}")


if __name__ == "__main__":
    main()
