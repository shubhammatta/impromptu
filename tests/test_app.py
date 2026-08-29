"""Headless smoke tests for the TUI, driven through textual's Pilot."""

import json

import httpx
import pyperclip
from textual.widgets import Button, Input, ListView, TextArea

import prompt_enhancer.app as app_module
from prompt_enhancer.app import ANSWER_PLACEHOLDER, INPUT_PLACEHOLDER, PromptEnhancerApp
from prompt_enhancer.config import LEVEL_SYSTEM_PROMPTS, SYSTEM_PROMPT
from prompt_enhancer.ollama import OllamaManager

MODEL = "qwen3.5:9b"

GENERATE_EVENTS = [
    {"response": "```markdown", "done": False},
    {"response": "\n# Role\nYou are a meticulous code reviewer.\n```", "done": False},
    {"response": "", "done": True},
]

QUESTIONS_JSON = '["Which programming language?", "Who is the target audience?"]'


def make_handler(
    ask_response: str = "[]",
    generate_bodies: list[dict] | None = None,
):
    """Mock Ollama: routes /api/generate on the `stream` flag (ask vs generate).

    `ask_response` is the raw reply to the question round. The sentinel
    "ndjson" makes the ask call receive streaming-style garbage, exercising the
    degrade-gracefully path.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": MODEL}]})
        if path == "/api/generate":
            payload = json.loads(request.content)
            if payload.get("stream") is False:
                if ask_response == "ndjson":
                    return httpx.Response(200, text='{"a": 1}\n{"b": 2}')
                return httpx.Response(200, json={"response": ask_response, "done": True})
            if generate_bodies is not None:
                generate_bodies.append(payload)
            return httpx.Response(
                200,
                text="\n".join(json.dumps(event) for event in GENERATE_EVENTS),
                headers={"content-type": "application/x-ndjson"},
            )
        return httpx.Response(404, json={"error": "not found"})

    return handler


def make_app(**kwargs) -> tuple[PromptEnhancerApp, OllamaManager]:
    manager = OllamaManager(
        "http://test",
        MODEL,
        transport=httpx.MockTransport(kwargs.pop("handler", make_handler())),
    )
    return PromptEnhancerApp(manager, **kwargs), manager


async def _wait_until(predicate, pilot, attempts: int = 150) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError("timed out waiting for application state")


# -- Direct generation (clarify off) ------------------------------------------------


async def test_boot_ready_generate_and_copy(monkeypatch):
    copied: list[str] = []
    monkeypatch.setattr(pyperclip, "copy", copied.append)

    app, _ = make_app(clarify=False)
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)

        prompt_input = app.query_one("#prompt-input", TextArea)
        assert not prompt_input.disabled

        prompt_input.text = "make me a code review prompt"
        await pilot.press("enter")
        await _wait_until(lambda: not app._generating, pilot)
        await _wait_until(lambda: app._in_rows_view, pilot)
        await pilot.pause(0.1)  # let the flush interval drain pending tokens

        # The refined result lands in the editable rows view, cleaned of fences.
        rows_text = "\n".join(app._result_rows)
        assert "meticulous code reviewer" in rows_text
        assert "```" not in rows_text
        rows_view = app.query_one("#output-rows", ListView)
        assert rows_view.display
        assert len(list(rows_view.children)) == len(app._result_rows)
        assert app.query_one("#output", TextArea).display is False

        app.action_copy_output()
        await _wait_until(lambda: bool(copied), pilot)
        assert "meticulous code reviewer" in copied[0]
        assert "```" not in copied[0]


async def test_empty_submit_is_rejected_and_input_kept():
    app, _ = make_app(clarify=False)
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)

        prompt_input = app.query_one("#prompt-input", TextArea)
        prompt_input.text = "   "
        await pilot.press("enter")
        await pilot.pause(0.1)

        assert not app._generating  # no generation was started
        assert prompt_input.text == "   "  # crude input not discarded


async def test_ctrl_q_shuts_down_cleanly():
    app, manager = make_app(clarify=False)
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        await pilot.press("ctrl+q")
    # Exiting the run_test context without an exception means action_quit
    # completed: workers were cancelled and the HTTP client was closed.
    assert manager._client is None


# -- Clarify is OFF by default --------------------------------------------------------


async def test_clarify_defaults_to_off(monkeypatch):
    monkeypatch.setattr(app_module, "CLARIFY_ENABLED", False)
    app, _ = make_app()  # no explicit clarify kwarg
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        assert app.clarify_enabled is False
        assert "clarify off" in app.sub_title


async def test_clarify_env_default_respected(monkeypatch):
    monkeypatch.setattr(app_module, "CLARIFY_ENABLED", True)
    app, _ = make_app()
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        assert app.clarify_enabled is True
        assert "clarify on" in app.sub_title


# -- Clarifying QnA -----------------------------------------------------------------


async def test_clarify_round_merges_answers_into_generation():
    generate_bodies: list[dict] = []
    app, _ = make_app(
        handler=make_handler(ask_response=QUESTIONS_JSON, generate_bodies=generate_bodies),
        clarify=True,
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)

        prompt_input = app.query_one("#prompt-input", TextArea)
        prompt_input.text = "write me a script"
        await pilot.press("enter")

        # Question 1
        await _wait_until(lambda: app._awaiting_answer, pilot)
        output = app.query_one("#output", TextArea)
        assert "(1/2)" in output.text and "Which programming language?" in output.text
        assert prompt_input.placeholder == ANSWER_PLACEHOLDER
        prompt_input.text = "Python 3.12"
        await pilot.press("enter")

        # Question 2
        await _wait_until(lambda: app._awaiting_answer, pilot)
        assert "(2/2)" in app.query_one("#output", TextArea).text
        prompt_input.text = "Backend engineers"
        await pilot.press("enter")

        await _wait_until(lambda: not app._generating, pilot)
        await _wait_until(lambda: app._in_rows_view, pilot)
        await pilot.pause(0.1)

        # The refined result is in the rows view, cleaned of fences.
        rows_text = "\n".join(app._result_rows)
        assert "meticulous code reviewer" in rows_text
        assert "```" not in rows_text
        assert app._awaiting_answer is False
        assert prompt_input.placeholder == INPUT_PLACEHOLDER

        # The generation request carried the crude prompt + merged Q/A.
        assert len(generate_bodies) == 1
        sent = generate_bodies[0]["prompt"]
        assert sent.startswith("write me a script")
        assert "## Clarifications from the user" in sent
        assert "Which programming language?" in sent and "Python 3.12" in sent
        assert "Who is the target audience?" in sent and "Backend engineers" in sent


async def test_clarify_skipped_questions_still_generate():
    generate_bodies: list[dict] = []
    app, _ = make_app(
        handler=make_handler(ask_response=QUESTIONS_JSON, generate_bodies=generate_bodies),
        clarify=True,
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)

        prompt_input = app.query_one("#prompt-input", TextArea)
        prompt_input.text = "write me a script"
        await pilot.press("enter")

        await _wait_until(lambda: app._awaiting_answer, pilot)
        await pilot.pause(0.6)  # deliberate empty skip must be past the grace window
        await pilot.press("enter")  # empty answer -> skip question 1
        await _wait_until(lambda: app._awaiting_answer, pilot)
        prompt_input.text = "CLI users"  # only answer question 2
        await pilot.press("enter")

        await _wait_until(lambda: not app._generating, pilot)
        assert len(generate_bodies) == 1
        sent = generate_bodies[0]["prompt"]
        assert "Which programming language?" not in sent  # skipped question dropped
        assert "Who is the target audience?" in sent
        assert "CLI users" in sent


async def test_double_enter_does_not_skip_a_question():
    """Regression: a fast second Enter after answering must never land as the
    NEXT question's answer (previously q2 got an empty answer and vanished)."""
    generate_bodies: list[dict] = []
    app, _ = make_app(
        handler=make_handler(ask_response=QUESTIONS_JSON, generate_bodies=generate_bodies),
        clarify=True,
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)

        prompt_input = app.query_one("#prompt-input", TextArea)
        prompt_input.text = "write me a script"
        await pilot.press("enter")

        # Question 1 — answer, then hammer Enter again before the worker resumes.
        await _wait_until(lambda: app._awaiting_answer, pilot)
        prompt_input.text = "Python 3.12"
        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.press("enter")

        # Question 2 must still appear and must still accept a real answer.
        await _wait_until(lambda: app._awaiting_answer, pilot)
        assert "(2/2)" in app.query_one("#output", TextArea).text
        prompt_input.text = "Backend engineers"
        await pilot.press("enter")
        await pilot.press("enter")  # stragglers must be ignored too

        await _wait_until(lambda: not app._generating, pilot)
        assert len(generate_bodies) == 1
        sent = generate_bodies[0]["prompt"]
        assert "Python 3.12" in sent  # q1 answer survived the extra Enters
        assert "Backend engineers" in sent  # q2 was answered once, not skipped
        # Both questions appear exactly once in the merged clarifications.
        assert sent.count("Which programming language?") == 1
        assert sent.count("Who is the target audience?") == 1


async def test_clarify_model_asks_nothing_proceeds_directly():
    generate_bodies: list[dict] = []
    app, _ = make_app(
        handler=make_handler(ask_response="[]", generate_bodies=generate_bodies),
        clarify=True,
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        prompt_input = app.query_one("#prompt-input", TextArea)
        prompt_input.text = "just do the thing"
        await pilot.press("enter")
        await _wait_until(lambda: not app._generating, pilot)

        assert len(generate_bodies) == 1
        assert generate_bodies[0]["prompt"] == "just do the thing"  # untouched


async def test_clarify_failure_degrades_to_plain_generation():
    generate_bodies: list[dict] = []
    app, _ = make_app(
        handler=make_handler(ask_response="ndjson", generate_bodies=generate_bodies),
        clarify=True,
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        prompt_input = app.query_one("#prompt-input", TextArea)
        prompt_input.text = "write me a script"
        await pilot.press("enter")
        await _wait_until(lambda: not app._generating, pilot)

        assert len(generate_bodies) == 1
        assert generate_bodies[0]["prompt"] == "write me a script"


async def test_ctrl_t_toggles_clarify_mode():
    generate_bodies: list[dict] = []
    app, _ = make_app(
        handler=make_handler(ask_response=QUESTIONS_JSON, generate_bodies=generate_bodies),
        clarify=True,
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        assert app.clarify_enabled is True

        await pilot.press("ctrl+t")
        assert app.clarify_enabled is False
        assert "clarify off" in app.sub_title

        await pilot.press("ctrl+t")
        assert app.clarify_enabled is True
        assert "clarify on" in app.sub_title


# -- Comprehensiveness level -----------------------------------------------------------


async def test_level_defaults_to_five_and_clamps():
    assert make_app(clarify=False)[0].level == 5
    assert make_app(clarify=False, level=0)[0].level == 1
    assert make_app(clarify=False, level=99)[0].level == 5

    app, _ = make_app(clarify=False)
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        assert str(app.query_one("#level-button", Button).label) == "Level 5"


async def test_level_cycles_and_wraps():
    app, _ = make_app(clarify=False, level=4)
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        level_button = app.query_one("#level-button", Button)

        app.action_cycle_level()
        assert app.level == 5
        assert str(level_button.label) == "Level 5"
        assert "L5 Exhaustive" in app.sub_title

        await pilot.press("ctrl+l")
        assert app.level == 1
        assert str(level_button.label) == "Level 1"
        assert "L1 Polish" in app.sub_title

        app.action_cycle_level()
        assert app.level == 2


async def test_level_selects_the_system_prompt_sent_to_ollama():
    generate_bodies: list[dict] = []
    app, _ = make_app(
        handler=make_handler(generate_bodies=generate_bodies), clarify=False, level=2
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        prompt_input = app.query_one("#prompt-input", TextArea)
        prompt_input.text = "a crude prompt"
        await pilot.press("enter")
        await _wait_until(lambda: not app._generating, pilot)

        assert len(generate_bodies) == 1
        assert generate_bodies[0]["system"] == LEVEL_SYSTEM_PROMPTS[2]


async def test_default_level_sends_the_original_system_prompt():
    generate_bodies: list[dict] = []
    app, _ = make_app(
        handler=make_handler(generate_bodies=generate_bodies), clarify=False
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        prompt_input = app.query_one("#prompt-input", TextArea)
        prompt_input.text = "a crude prompt"
        await pilot.press("enter")
        await _wait_until(lambda: not app._generating, pilot)

        assert generate_bodies[0]["system"] == SYSTEM_PROMPT


# -- Result rows: delete & inline edit -------------------------------------------------


async def _generate_result(app: PromptEnhancerApp, pilot, text: str) -> None:
    await _wait_until(lambda: app._booted, pilot)
    prompt_input = app.query_one("#prompt-input", TextArea)
    prompt_input.text = text
    await pilot.press("enter")
    await _wait_until(lambda: not app._generating, pilot)
    await _wait_until(lambda: app._in_rows_view, pilot)
    await pilot.pause(0.1)


async def test_row_delete_updates_rows_and_copy(monkeypatch):
    copied: list[str] = []
    monkeypatch.setattr(pyperclip, "copy", copied.append)
    app, _ = make_app(clarify=False)
    async with app.run_test() as pilot:
        await _generate_result(app, pilot, "make me a code review prompt")
        original_count = len(app._result_rows)
        assert original_count >= 2

        deleted_row = app._result_rows[0]
        first_delete = list(app.query(".row-del-btn"))[0]
        first_delete.press()
        await pilot.pause(0.1)

        assert len(app._result_rows) == original_count - 1
        assert deleted_row not in app._result_rows
        rows_view = app.query_one("#output-rows", ListView)
        assert len(list(rows_view.children)) == original_count - 1

        app.action_copy_output()
        await _wait_until(lambda: bool(copied), pilot)
        assert deleted_row not in copied[0]


async def test_row_inline_edit_commits_and_copies(monkeypatch):
    copied: list[str] = []
    monkeypatch.setattr(pyperclip, "copy", copied.append)
    app, _ = make_app(clarify=False)
    async with app.run_test() as pilot:
        await _generate_result(app, pilot, "make me a code review prompt")
        original_row = app._result_rows[0]

        first_edit = list(app.query(".row-edit-btn"))[0]
        first_edit.press()
        await pilot.pause(0.1)

        editor = app.query_one(".row-input", Input)
        assert editor.value == original_row
        editor.value = "totally rewritten line"
        await pilot.press("enter")  # commits via the Enter binding

        assert app._edit_index is None
        assert app._result_rows[0] == "totally rewritten line"
        assert len(list(app.query(".row-input"))) == 0  # editor gone

        app.action_copy_output()
        await _wait_until(lambda: bool(copied), pilot)
        assert "totally rewritten line" in copied[0]
        assert original_row not in copied[0]


async def test_row_edit_escape_cancels_without_changes():
    app, _ = make_app(clarify=False)
    async with app.run_test() as pilot:
        await _generate_result(app, pilot, "make me a code review prompt")
        original_row = app._result_rows[0]

        first_edit = list(app.query(".row-edit-btn"))[0]
        first_edit.press()
        await pilot.pause(0.1)

        editor = app.query_one(".row-input", Input)
        editor.value = "this must not be kept"
        await pilot.press("escape")

        assert app._edit_index is None
        assert app._result_rows[0] == original_row


# -- Prompt history (Up/Down traversal) ---------------------------------------------------


async def test_up_down_traverses_submitted_prompts():
    app, _ = make_app(clarify=False)
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        pi = app.query_one("#prompt-input", TextArea)
        for text in ("prompt one", "prompt two"):
            pi.text = text
            await pilot.press("enter")
            await _wait_until(lambda: not app._generating, pilot)
            await _wait_until(lambda: app._in_rows_view, pilot)
            await pilot.pause(0.1)

        pi.text = ""
        await pilot.press("up")
        assert pi.text == "prompt two"
        await pilot.press("up")
        assert pi.text == "prompt one"
        await pilot.press("up")
        assert pi.text == "prompt one"  # stays at the oldest entry
        await pilot.press("down")
        assert pi.text == "prompt two"
        await pilot.press("down")
        assert pi.text == ""  # the pre-traversal draft is restored


async def test_browsed_history_entry_resubmits_and_dedupes():
    generate_bodies: list[dict] = []
    app, _ = make_app(
        handler=make_handler(generate_bodies=generate_bodies), clarify=False
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        pi = app.query_one("#prompt-input", TextArea)
        pi.text = "prompt one"
        await pilot.press("enter")
        await _wait_until(lambda: not app._generating, pilot)
        await _wait_until(lambda: app._in_rows_view, pilot)
        await pilot.pause(0.1)

        await pilot.press("up")  # recall, then submit unchanged
        assert pi.text == "prompt one"
        await pilot.press("enter")
        await _wait_until(lambda: len(generate_bodies) == 2, pilot)

        assert generate_bodies[1]["prompt"] == "prompt one"
        assert app.history.entries == ["prompt one"]  # no consecutive duplicate


async def test_history_persists_across_restart(tmp_path):
    hist = tmp_path / "history.txt"
    app1, _ = make_app(clarify=False, history_path=str(hist))
    async with app1.run_test() as pilot:
        await _wait_until(lambda: app1._booted, pilot)
        pi = app1.query_one("#prompt-input", TextArea)
        pi.text = "remembered prompt"
        await pilot.press("enter")
        await _wait_until(lambda: not app1._generating, pilot)

    app2, _ = make_app(clarify=False, history_path=str(hist))
    async with app2.run_test() as pilot:
        await _wait_until(lambda: app2._booted, pilot)
        pi = app2.query_one("#prompt-input", TextArea)
        pi.text = ""
        await pilot.press("up")
        assert pi.text == "remembered prompt"


async def test_up_during_clarify_answer_does_not_traverse_history():
    app, _ = make_app(
        handler=make_handler(ask_response=QUESTIONS_JSON), clarify=True
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        pi = app.query_one("#prompt-input", TextArea)
        pi.text = "write me a script"
        await pilot.press("enter")
        await _wait_until(lambda: app._awaiting_answer, pilot)

        pi.text = "my partial answer"
        await pilot.press("up")  # must not dump a past prompt into the answer
        assert pi.text == "my partial answer"


# -- Multi-line prompt input ---------------------------------------------------------


async def test_shift_and_alt_enter_insert_newlines_and_submit_multiline():
    generate_bodies: list[dict] = []
    app, _ = make_app(
        handler=make_handler(generate_bodies=generate_bodies), clarify=False
    )
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        pi = app.query_one("#prompt-input", TextArea)

        pi.text = "part one"
        pi.move_cursor(pi.document.end)
        await pilot.press("shift+enter")
        assert pi.text == "part one\n"

        pi.insert("part two")  # as if typed after the newline
        pi.move_cursor(pi.document.end)
        await pilot.press("alt+enter")
        pi.insert("part three")
        assert pi.text == "part one\npart two\npart three"

        await pilot.press("enter")  # Enter still submits the whole thing
        await _wait_until(lambda: len(generate_bodies) == 1, pilot)
        assert generate_bodies[0]["prompt"] == "part one\npart two\npart three"


async def test_up_moves_cursor_inside_multiline_text_instead_of_history():
    app, _ = make_app(clarify=False)
    async with app.run_test() as pilot:
        await _wait_until(lambda: app._booted, pilot)
        pi = app.query_one("#prompt-input", TextArea)

        pi.text = "prompt one"
        await pilot.press("enter")
        await _wait_until(lambda: not app._generating, pilot)
        await _wait_until(lambda: app._in_rows_view, pilot)
        await pilot.pause(0.1)

        pi.text = "first line\nsecond line"
        pi.move_cursor(pi.document.end)  # cursor on the last line
        await pilot.press("up")
        assert pi.cursor_location[0] == 0  # moved up a line…
        assert pi.text == "first line\nsecond line"  # …no history recall

        await pilot.press("down")
        assert pi.cursor_location[0] == 1
        assert pi.text == "first line\nsecond line"

        await pilot.press("up")  # cursor hop back to the first line…
        assert pi.cursor_location[0] == 0
        assert pi.text == "first line\nsecond line"
        await pilot.press("up")  # …next Up on the first line recalls history
        assert pi.text == "prompt one"
