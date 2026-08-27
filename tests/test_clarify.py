from prompt_enhancer.clarify import (
    build_final_prompt,
    build_question_prompt,
    parse_questions,
)


def test_parses_plain_json_array():
    assert parse_questions('["What language?", "Which audience?"]') == [
        "What language?",
        "Which audience?",
    ]


def test_parses_json_inside_reasoning_and_fences():
    raw = "<think>Two things are unclear.</think>\n```json\n[\"Target language?\"]\n```"
    assert parse_questions(raw) == ["Target language?"]


def test_empty_array_means_no_questions():
    assert parse_questions("[]") == []


def test_caps_at_two_questions():
    assert parse_questions('["a?", "b?", "c?", "d?"]') == ["a?", "b?"]


def test_ignores_non_string_array_items():
    assert parse_questions('["Real question?", 42, ["nested"]]') == ["Real question?"]


def test_fallback_to_question_lines_without_json():
    raw = "1. What is the target language?\n2. Who reads the output?\nSome note without a question mark."
    assert parse_questions(raw) == ["What is the target language?", "Who reads the output?"]


def test_garbage_yields_no_questions():
    assert parse_questions("I have absolutely no idea what you mean.") == []


def test_dedupes_case_insensitively_and_collapses_whitespace():
    raw = '["Same  question?", "same question?", "Other?"]'
    assert parse_questions(raw) == ["Same question?", "Other?"]


def test_question_prompt_contains_crude_text():
    prompt = build_question_prompt("make me a tool")
    assert "make me a tool" in prompt
    assert "JSON array" in prompt


def test_final_prompt_without_answers_is_unchanged():
    assert build_final_prompt("crude", []) == "crude"


def test_final_prompt_appends_clarifications():
    out = build_final_prompt("crude", [("Q1?", "A1"), ("Q2?", "A2")])
    assert out.startswith("crude")
    assert "## Clarifications from the user" in out
    assert "- Q: Q1?" in out
    assert "  A: A1" in out
