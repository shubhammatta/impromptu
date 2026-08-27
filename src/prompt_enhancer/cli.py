"""Command-line entry point for Prompt Enhancer."""

from __future__ import annotations

import argparse

from . import __version__
from .app import PromptEnhancerApp
from .config import DEFAULT_LEVEL, HISTORY_PATH, LEVEL_NAMES, MODEL, OLLAMA_HOST
from .ollama import OllamaManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-enhancer",
        description="Turn crude prompts into engineered ones with a local Ollama model.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Ollama model to use (default: {MODEL}; env PROMPT_ENHANCER_MODEL).",
    )
    parser.add_argument(
        "--host",
        default=OLLAMA_HOST,
        help=f"Ollama base URL (default: {OLLAMA_HOST}; env PROMPT_ENHANCER_HOST).",
    )
    parser.add_argument(
        "--clarify",
        dest="clarify",
        action="store_true",
        default=None,
        help="Ask up to 2 clarifying questions before enhancing (default: off).",
    )
    parser.add_argument(
        "--no-clarify",
        dest="clarify",
        action="store_false",
        help="Skip the clarifying-question round (the default).",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=sorted(LEVEL_NAMES),
        default=None,
        help=(
            "How comprehensive the enhancement should be, 1-5: "
            + ", ".join(f"{n}={LEVEL_NAMES[n]}" for n in sorted(LEVEL_NAMES))
            + f" (default: {DEFAULT_LEVEL}; env PROMPT_ENHANCER_LEVEL)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manager = OllamaManager(host=args.host, model=args.model)
    PromptEnhancerApp(
        manager, clarify=args.clarify, level=args.level, history_path=HISTORY_PATH
    ).run()


if __name__ == "__main__":  # pragma: no cover
    main()
