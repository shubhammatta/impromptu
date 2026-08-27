"""Post-processing for raw model output -> the clean, copyable prompt."""

from __future__ import annotations

import re

# Qwen3-family models emit a <think>…</think> reasoning block (Ollama routes it
# to the `thinking` NDJSON field on newer versions, but strip defensively in
# case it leaks into `response`). An unterminated block (e.g. truncation) is
# treated as all-think and dropped too.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

# The system prompt demands "ONLY the final usable prompt inside a markdown
# block". If the whole reply is one fenced block, unwrap it so the TextArea and
# the clipboard hold the bare prompt rather than fence syntax.
_FENCE_RE = re.compile(r"\A```[ \t]*[\w+-]*[ \t]*\r?\n(.*?)\r?\n?```[ \t]*\r?\n?\Z", re.DOTALL)


def strip_think(raw: str) -> str:
    """Remove Qwen-style <think>…</think> reasoning traces (terminated or not)."""
    return _THINK_RE.sub("", raw).strip()


def extract_prompt(raw: str) -> str:
    """Strip reasoning traces and a wrapping markdown fence from a generation."""
    text = strip_think(raw)
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    return text
