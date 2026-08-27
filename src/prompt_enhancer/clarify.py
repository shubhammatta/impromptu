"""Clarifying QnA: build the question ask, parse the model's questions, merge answers.

Pure functions only — the model call lives in OllamaManager.ask() and the
conversation flow lives in the app, so this module is trivially unit-testable.
"""

from __future__ import annotations

import json

from .config import MAX_QUESTIONS, QUESTION_SYSTEM_PROMPT
from .text import strip_think

__all__ = [
    "QUESTION_SYSTEM_PROMPT",
    "build_question_prompt",
    "build_final_prompt",
    "parse_questions",
]


def build_question_prompt(crude: str) -> str:
    return (
        "Crude prompt from the user:\n"
        "<crude>\n"
        f"{crude}\n"
        "</crude>\n\n"
        f"Ask up to {MAX_QUESTIONS} clarifying questions as a JSON array "
        "(an empty array if none are needed)."
    )


def parse_questions(raw: str, max_questions: int = MAX_QUESTIONS) -> list[str]:
    """Extract up to `max_questions` questions from a model reply.

    Accepts a JSON array (fenced or not, possibly wrapped in reasoning traces)
    and falls back to question-looking lines if no JSON is present. Anything
    unparseable yields [] — the caller proceeds without clarification.
    """
    text = strip_think(raw)
    questions: list[str] = []

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            questions = [item.strip() for item in data if isinstance(item, str)]

    if not questions:  # fallback: numbered or bare question lines
        questions = [
            line.strip().lstrip("0123456789.:-) ")
            for line in text.splitlines()
            if line.strip().endswith("?")
        ]

    # Normalize whitespace inside questions, drop empties and duplicates.
    seen: set[str] = set()
    cleaned: list[str] = []
    for question in questions:
        question = " ".join(question.split())
        if question and question.lower() not in seen:
            seen.add(question.lower())
            cleaned.append(question)
    return cleaned[:max_questions]


def build_final_prompt(crude: str, answers: list[tuple[str, str]]) -> str:
    """Merge the crude prompt with collected Q/A pairs into the generation input."""
    if not answers:
        return crude
    lines = [crude, "", "## Clarifications from the user", ""]
    for question, answer in answers:
        lines.append(f"- Q: {question}")
        lines.append(f"  A: {answer}")
    return "\n".join(lines)
