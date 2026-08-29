"""Static configuration for Prompt Enhancer.

Every value can be overridden from the environment so the app works against
non-default Ollama hosts/models without code changes.
"""

from __future__ import annotations

import os

# --- Endpoints -----------------------------------------------------------------
OLLAMA_HOST = os.environ.get("PROMPT_ENHANCER_HOST", "http://localhost:11434")
HEALTH_PATH = "/api/tags"  # doubles as the health-check endpoint (cheap, always 200 when up)
PULL_PATH = "/api/pull"
GENERATE_PATH = "/api/generate"

# --- Model ---------------------------------------------------------------------
MODEL = os.environ.get("PROMPT_ENHANCER_MODEL", "qwen3.5:9b")

# --- Lifecycle timings ----------------------------------------------------------
POLL_INTERVAL = 0.5  # seconds between readiness probes after spawning `ollama serve`
STARTUP_TIMEOUT = 15.0  # seconds to wait for the daemon to come up
HEALTH_TIMEOUT = 1.0  # per-request timeout for health probes (connect capped below)

# Long read timeouts: a cold 9B model can spend minutes paging weights in before
# the first token/NDJSON line arrives, and model pulls are bandwidth-bound.
GENERATE_READ_TIMEOUT = 300.0
PULL_READ_TIMEOUT = 120.0

# --- Generation options ---------------------------------------------------------
TEMPERATURE = 0.7
TOP_P = 0.9
NUM_CTX = 8192

# Qwen3.5 is a thinking model: by default it streams a separate `thinking` field
# of chain-of-thought before the answer — for a short prompt that is often
# thousands of tokens (a minute of wall time at ~53 tok/s) of invisible latency
# the UI cannot show. Thinking is therefore OFF by default; opt back in with
# PROMPT_ENHANCER_THINK=1.
ENABLE_THINKING = os.environ.get("PROMPT_ENHANCER_THINK", "").lower() in {
    "1",
    "true",
    "yes",
}

# --- Clarifying QnA ---------------------------------------------------------------
# Ask up to N clarifying questions before enhancing when the mode is enabled.
MAX_QUESTIONS = 2
# Default switch (CLI --clarify/--no-clarify or Ctrl+T override it per run).
# OFF by default: opt in per run with Ctrl+T / --clarify / PROMPT_ENHANCER_CLARIFY=1.
CLARIFY_ENABLED = os.environ.get("PROMPT_ENHANCER_CLARIFY", "0").lower() in {
    "1",
    "true",
    "yes",
}
ASK_READ_TIMEOUT = 120.0  # question round-trip; short output, but cold loads happen

QUESTION_SYSTEM_PROMPT = """You are a Requirements Analyst helping an Expert Prompt Engineer.
Given a crude user prompt, decide whether clarifying questions would materially improve the final engineered prompt.
If they would, ask the fewest, highest-leverage questions needed (at most 2). If the prompt is already clear enough, ask nothing.

Respond with ONLY a JSON array of short question strings. No preamble, no commentary.
Examples:
[]
["Which programming language and version should the solution target?", "Who is the intended audience for the output?"]"""

# --- Process management ----------------------------------------------------------
# Path of the `ollama` binary. Empty -> resolved via shutil.which at spawn time.
OLLAMA_BIN = os.environ.get("PROMPT_ENHANCER_OLLAMA_BIN", "")
# If "1"/"true"/"yes", a daemon WE spawned is terminated on app quit. By default
# the detached daemon is left running (it is a useful system service, and
# `start_new_session=True` guarantees it never blocks or zombies our shell).
STOP_OLLAMA_ON_EXIT = os.environ.get("PROMPT_ENHANCER_STOP_OLLAMA_ON_EXIT", "").lower() in {
    "1",
    "true",
    "yes",
}
LOG_DIR = os.path.join(os.path.expanduser("~"), ".cache", "prompt-enhancer")
SERVE_LOG_PATH = os.path.join(LOG_DIR, "ollama-serve.log")

# --- Prompt history ----------------------------------------------------------------
# Submitted crude prompts, one per line, traversed with Up/Down like a shell.
HISTORY_MAX_ENTRIES = 200
HISTORY_PATH = os.environ.get(
    "PROMPT_ENHANCER_HISTORY", os.path.join(LOG_DIR, "history.txt")
)

# --- Prompt engineering ----------------------------------------------------------
SYSTEM_PROMPT = """You are an Expert Prompt Engineer and AI Architect.
Your task is to transform crude, unstructured user inputs into comprehensive, high-performance prompts optimized for modern LLMs.

Structure the output with:
- Role & Persona
- Task & Objectives
- Context & Constraints
- Expected Output Format (e.g., Markdown, JSON, structured sections)
- Chain-of-Thought / reasoning guidelines if applicable

Output ONLY the final usable prompt inside a markdown block. Do not include conversational preambles, meta-commentary, or pleasantries."""

# --- Comprehensiveness level ------------------------------------------------------
# 1 = light polish … 5 = exhaustive engineering (the original full config above).
_raw_level = os.environ.get("PROMPT_ENHANCER_LEVEL", "5")
try:
    DEFAULT_LEVEL = min(5, max(1, int(_raw_level)))
except ValueError:
    DEFAULT_LEVEL = 5

LEVEL_NAMES = {1: "Polish", 2: "Focused", 3: "Structured", 4: "Detailed", 5: "Exhaustive"}

# Level 5 is the verbatim original system prompt; 1-4 are progressively lighter.
LEVEL_SYSTEM_PROMPTS: dict[int, str] = {
    1: """You are an Expert Prompt Engineer.
Polish the user's prompt: correct grammar, sharpen vague wording, and make the intent explicit.
Keep the prompt essentially the same length and structure — do not add new sections, requirements, or constraints.

Output ONLY the polished prompt inside a markdown block. No preamble, meta-commentary, or pleasantries.""",
    2: """You are an Expert Prompt Engineer.
Improve the user's prompt: correct grammar and wording, make the intent explicit, and add only the missing essentials — a clear task statement, the target audience, and the expected output format.
Keep it compact; do not invent requirements.

Output ONLY the improved prompt inside a markdown block. No preamble, meta-commentary, or pleasantries.""",
    3: """You are an Expert Prompt Engineer.
Restructure the user's prompt into clearly labeled sections: Role, Task, Context & Constraints, and Expected Output Format.
Make every requirement concrete and testable. Keep it as short as the structure allows.

Output ONLY the restructured prompt inside a markdown block. No preamble, meta-commentary, or pleasantries.""",
    4: """You are an Expert Prompt Engineer and AI Architect.
Transform the user's input into a comprehensive prompt structured with:
- Role & Persona
- Task & Objectives
- Context & Constraints
- Expected Output Format
- Success criteria and edge cases to handle
- 1-2 brief examples of a good result, if they add clarity

Output ONLY the final usable prompt inside a markdown block. No conversational preambles, meta-commentary, or pleasantries.""",
    5: SYSTEM_PROMPT,
}
assert set(LEVEL_SYSTEM_PROMPTS) == set(LEVEL_NAMES) == set(range(1, 6))
