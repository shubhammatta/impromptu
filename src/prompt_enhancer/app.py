"""The Prompt Enhancer Textual application."""

from __future__ import annotations

import asyncio
import time
from functools import partial

import pyperclip
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    ListItem,
    ListView,
    Static,
    TextArea,
)

from .clarify import build_final_prompt, build_question_prompt, parse_questions
from .config import (
    CLARIFY_ENABLED,
    DEFAULT_LEVEL,
    HISTORY_MAX_ENTRIES,
    LEVEL_NAMES,
    LEVEL_SYSTEM_PROMPTS,
    QUESTION_SYSTEM_PROMPT,
    STOP_OLLAMA_ON_EXIT,
)
from .history import PromptHistory
from .ollama import OllamaError, OllamaManager
from .text import extract_prompt

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
INPUT_PLACEHOLDER = "Enter crude prompt..."
ANSWER_PLACEHOLDER = "Type an answer (or press Enter to skip) …"
LOADING_HINT = "(first run loads the model into memory; this can take a while)"
# EMPTY Enter presses (skip attempts) within this window of a question being
# presented are ignored: they are key-repeat/bounce from answering the previous
# question and must never skip the new one. Deliberate skips come after a human
# reading pause; typed answers are never empty and bypass the window entirely.
ANSWER_GRACE_SECONDS = 0.5


class HistoryInput(Input):
    """Main prompt input with shell-style Up/Down history traversal."""

    BINDINGS = [
        Binding("up", "history_prev", show=False),
        Binding("down", "history_next", show=False),
    ]

    def action_history_prev(self) -> None:
        app = self.app
        assert isinstance(app, PromptEnhancerApp)
        if app._awaiting_answer:
            return  # Up during a question would shove a prompt into the answer
        entry = app.history.older(self.value)
        if entry is not None:
            self.value = entry
            self.cursor_position = len(entry)

    def action_history_next(self) -> None:
        app = self.app
        assert isinstance(app, PromptEnhancerApp)
        if app._awaiting_answer:
            return
        entry = app.history.newer()
        if entry is not None:
            self.value = entry
            self.cursor_position = len(entry)


class PromptEnhancerApp(App[None]):
    """A single-screen tool: crude prompt in (optionally refined by a short QnA,
    at a chosen comprehensiveness level), engineered prompt out, edit, copy, go."""

    TITLE = "⚡ Prompt Enhancer"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("enter", "submit", "Submit", priority=True),
        Binding("ctrl+c", "copy_output", "Copy", priority=True),
        Binding("ctrl+t", "toggle_clarify", "Clarify", priority=True),
        Binding("ctrl+l", "cycle_level", "Level", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        manager: OllamaManager,
        *,
        clarify: bool | None = None,
        level: int | None = None,
        history_path: str | None = None,
    ) -> None:
        super().__init__()
        self.manager = manager
        self.clarify_enabled = CLARIFY_ENABLED if clarify is None else clarify
        self.level = DEFAULT_LEVEL if level is None else min(5, max(1, int(level)))
        self.history = PromptHistory(history_path, max_entries=HISTORY_MAX_ENTRIES)
        # --- state machine ---
        self._booted = False           # boot sequence finished successfully
        self._generating = False       # an enhancement worker is in flight
        self._awaiting_answer = False  # a clarifying question is on screen
        self._status_mode = True       # TextArea shows the boot log vs. generated output
        self._status_lines: list[str] = []
        self._transient_last = False   # last status line is a progress line (replaceable)
        self._spinner_active = False
        self._spinner_frame = 0
        self._loading_message = ""
        self._stream_parts: list[str] = []
        self._stream_started = False   # first streamed token wipes placeholder text
        self._pending: list[str] = []  # tokens awaiting the flush interval
        self._answer_event: asyncio.Event | None = None
        self._last_answer = ""
        self._question_shown_at = 0.0  # monotonic; anchors the Enter grace window
        # --- editable result rows ---
        self._in_rows_view = False
        self._result_rows: list[str] = []
        self._row_statics: list[Static] = []
        self._edit_index: int | None = None
        self._edit_editor: Input | None = None
        self._edit_static: Static | None = None

    # -- Layout ----------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main"):
            # show_cursor=False: a read-only pane has no business showing a
            # blinking cursor — it parked on line 1 and read as flickering text.
            yield TextArea(
                id="output",
                read_only=True,
                soft_wrap=True,
                show_line_numbers=False,
                show_cursor=False,
            )
            yield ListView(id="output-rows")
        with Horizontal(id="input-bar"):
            yield HistoryInput(placeholder=INPUT_PLACEHOLDER, id="prompt-input")
            yield Button(f"Level {self.level}", id="level-button")
            yield Button("Copy", variant="primary", id="copy-button")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick_spinner)
        self.set_interval(0.08, self._flush_stream)
        output = self.query_one("#output", TextArea)
        output.border_title = "system status"
        self._boot()

    # -- Boot ------------------------------------------------------------------

    @work(group="boot", exclusive=True)
    async def _boot(self) -> None:
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.disabled = True
        self._push_status("◌ Booting Prompt Enhancer …")
        try:
            await self.manager.ensure_ready(on_event=self._push_status, on_progress=self._push_progress)
        except OllamaError as exc:
            self._push_status(f"✗ {exc}")
            self.notify(str(exc), title="Startup failed", severity="error", timeout=10)
            return
        except Exception as exc:  # keep the TUI alive no matter what boot throws
            self._push_status(f"✗ Unexpected startup error: {exc!r}")
            self.notify(f"Unexpected startup error: {exc!r}", severity="error", timeout=10)
            return
        self._booted = True
        self._push_status(f"✓ Ready — enter a crude prompt below. Model: {self.manager.model}")
        self._refresh_sub_title()
        prompt_input.disabled = False
        prompt_input.focus()

    def _refresh_sub_title(self) -> None:
        clarify_state = "on" if self.clarify_enabled else "off"
        self.sub_title = (
            f"{self.manager.model} @ {self.manager.host} · "
            f"L{self.level} {LEVEL_NAMES[self.level]} · clarify {clarify_state}"
        )

    # -- Status log rendering ---------------------------------------------------

    def _push_status(self, line: str) -> None:
        if not self._status_mode:
            return
        self._status_lines.append(line)
        self._transient_last = False
        self._render_status()

    def _push_progress(self, line: str) -> None:
        if not self._status_mode:
            return
        if self._transient_last and self._status_lines:
            self._status_lines[-1] = line
        else:
            self._status_lines.append(line)
        self._transient_last = True
        self._render_status()

    def _render_status(self) -> None:
        self.query_one("#output", TextArea).text = "\n".join(self._status_lines)

    # -- View switching (status/stream pane vs. editable rows) -------------------

    def _show_text_view(self) -> None:
        self._in_rows_view = False
        self._end_row_edit()
        self.query_one("#output-rows", ListView).display = False
        self.query_one("#output", TextArea).display = True

    def _show_rows(self, text: str) -> None:
        self._result_rows = text.split("\n")
        self._rebuild_rows()
        self._in_rows_view = True
        self.query_one("#output", TextArea).display = False
        rows_view = self.query_one("#output-rows", ListView)
        rows_view.display = True
        rows_view.border_title = f"refined prompt — editable ({len(self._result_rows)} lines)"

    # -- Result row editing ---------------------------------------------------------

    def _rebuild_rows(self) -> None:
        list_view = self.query_one("#output-rows", ListView)
        list_view.clear()
        self._row_statics = []
        for text in self._result_rows:
            static = Static(text, classes="row-text")
            edit_button = Button("✏", classes="row-btn row-edit-btn")
            delete_button = Button("✖", classes="row-btn row-del-btn")
            list_view.append(ListItem(Horizontal(static, edit_button, delete_button)))
            self._row_statics.append(static)

    def _row_index(self, button: Button) -> int | None:
        node = button.parent
        while node is not None and not isinstance(node, ListItem):
            node = node.parent
        if node is None:
            return None
        list_view = self.query_one("#output-rows", ListView)
        for index, child in enumerate(list_view.children):
            if child is node:
                return index
        return None

    @on(Button.Pressed, ".row-edit-btn")
    def _on_row_edit_pressed(self, event: Button.Pressed) -> None:
        index = self._row_index(event.button)
        if index is not None and self._in_rows_view:
            self._begin_row_edit(index)

    @on(Button.Pressed, ".row-del-btn")
    def _on_row_delete_pressed(self, event: Button.Pressed) -> None:
        index = self._row_index(event.button)
        if index is None or not self._in_rows_view:
            return
        self._end_row_edit()
        del self._result_rows[index]
        del self._row_statics[index]
        list_view = self.query_one("#output-rows", ListView)
        list_view.remove_items([index])
        rows_view_title = f"refined prompt — editable ({len(self._result_rows)} lines)"
        list_view.border_title = rows_view_title

    def _begin_row_edit(self, index: int) -> None:
        self._end_row_edit()  # only one editor at a time
        list_view = self.query_one("#output-rows", ListView)
        item = list(list_view.children)[index]
        static = self._row_statics[index]
        editor = Input(value=self._result_rows[index], classes="row-input")
        self._edit_index = index
        self._edit_editor = editor
        self._edit_static = static
        static.display = False
        item.mount(editor)
        self.call_after_refresh(editor.focus)

    def _commit_row_edit(self) -> None:
        if self._edit_editor is None or self._edit_index is None:
            return
        value = self._edit_editor.value
        self._result_rows[self._edit_index] = value
        if self._edit_static is not None:
            self._edit_static.update(value)
        self._end_row_edit()

    def _end_row_edit(self) -> None:
        if self._edit_editor is not None:
            self._edit_editor.remove()
        if self._edit_static is not None:
            self._edit_static.display = True
        self._edit_index = None
        self._edit_editor = None
        self._edit_static = None

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape" and self._edit_index is not None:
            self._end_row_edit()

    # -- Level / clarify toggles -----------------------------------------------------

    @on(Button.Pressed, "#level-button")
    def _on_level_pressed(self, event: Button.Pressed) -> None:
        self.action_cycle_level()

    def action_cycle_level(self) -> None:
        self.level = self.level % 5 + 1
        self.query_one("#level-button", Button).label = f"Level {self.level}"
        self._refresh_sub_title()
        self.notify(
            f"Level {self.level} — {LEVEL_NAMES[self.level]}.",
            title="Comprehensiveness",
            timeout=2,
        )

    def action_toggle_clarify(self) -> None:
        self.clarify_enabled = not self.clarify_enabled
        self._refresh_sub_title()
        state = "enabled" if self.clarify_enabled else "disabled"
        self.notify(f"Clarifying questions {state}.", title="Clarify", timeout=2)

    # -- Submission & enhancement -------------------------------------------------

    def action_submit(self) -> None:
        prompt_input = self.query_one("#prompt-input", Input)
        if self._awaiting_answer:
            # Latch: only the FIRST Enter per question is accepted. A repeat
            # press must never fall through and answer the NEXT question with
            # an empty string before it is even presented.
            answer = prompt_input.value.strip()
            if (
                prompt_input.disabled
                or self._answer_event is None
                or self._answer_event.is_set()
                or (not answer and time.monotonic() - self._question_shown_at < ANSWER_GRACE_SECONDS)
            ):
                return
            self._last_answer = answer
            prompt_input.value = ""
            self._answer_event.set()
            self.history.reset_browse()
            return
        if self._edit_index is not None:
            # Enter while a row editor is open commits it (priority binding
            # would otherwise swallow the editor's own submit).
            self._commit_row_edit()
            return
        if self._generating:
            self.notify("Still enhancing the previous prompt …", title="Busy", severity="warning")
            return
        if not self._booted:
            self.notify("Ollama is still starting up …", title="Please wait", severity="warning")
            return
        crude = prompt_input.value.strip()
        if not crude:
            self.notify("Type a crude prompt first.", title="Empty input", severity="warning")
            return
        self.history.add(crude)
        prompt_input.value = ""
        prompt_input.disabled = True
        self._generating = True
        self._status_mode = False
        self._stream_parts = []
        self._pending = []
        self._stream_started = False
        self._show_text_view()
        output = self.query_one("#output", TextArea)
        output.border_title = f"response — {self.manager.model}"
        self._start_spinner(f"Enhancing with {self.manager.model} …")
        self._enhance(crude)

    @work(group="generate", exclusive=True)
    async def _enhance(self, crude: str) -> None:
        try:
            answers: list[tuple[str, str]] = []
            if self.clarify_enabled:
                try:
                    questions = await self._obtain_questions(crude)
                except OllamaError as exc:
                    # Clarification is best-effort: never trap the user behind it.
                    questions = []
                    self.notify(
                        f"Clarification skipped: {exc}", title="Clarify", severity="warning", timeout=6
                    )
                if questions:
                    for index, question in enumerate(questions, start=1):
                        self._present_question(question, index, len(questions))
                        answer = await self._collect_answer()
                        if answer:
                            answers.append((question, answer))
                    self._start_spinner("Enhancing with your clarifications …")
                else:
                    self._start_spinner("No clarification needed — enhancing …")

            final = await self.manager.generate(
                build_final_prompt(crude, answers),
                on_token=self._on_token,
                system=LEVEL_SYSTEM_PROMPTS[self.level],
            )
        except asyncio.CancelledError:
            raise
        except OllamaError as exc:
            self._finish_generation(error=str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected in-UI
            self._finish_generation(error=repr(exc))
            return
        self._finish_generation(result=final)

    async def _obtain_questions(self, crude: str) -> list[str]:
        self._start_spinner("Reading your prompt — preparing clarifying questions …")
        raw = await self.manager.ask(
            build_question_prompt(crude), system=QUESTION_SYSTEM_PROMPT
        )
        questions = parse_questions(raw)
        self._spinner_active = False
        return questions

    def _present_question(self, question: str, index: int, total: int) -> None:
        output = self.query_one("#output", TextArea)
        output.border_title = "clarification"
        output.text = (
            f"To tailor the prompt, please answer ({index}/{total}):\n\n"
            f"  ? {question}\n\n"
            "Type an answer and press Enter — or leave it empty to skip."
        )

    async def _collect_answer(self) -> str:
        """Hand control to the Input until the user answers; empty answer = skip."""
        self._awaiting_answer = True
        self._answer_event = asyncio.Event()
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.disabled = False
        prompt_input.placeholder = ANSWER_PLACEHOLDER
        self._question_shown_at = time.monotonic()
        prompt_input.focus()
        try:
            await self._answer_event.wait()
        finally:
            # Take the input away between questions so stray keys/Enters in
            # the transition window can never be misrouted into the next one.
            self._awaiting_answer = False
            self._answer_event = None
            prompt_input.disabled = True
            prompt_input.placeholder = INPUT_PLACEHOLDER
        return self._last_answer

    def _finish_generation(self, result: str | None = None, error: str | None = None) -> None:
        self._generating = False
        self._spinner_active = False
        self._awaiting_answer = False
        self._pending.clear()
        if error is not None:
            self._show_text_view()
            output = self.query_one("#output", TextArea)
            output.text = f"✗ Generation failed: {error}"
            output.border_title = "error"
            self.notify(error, title="Generation failed", severity="error", timeout=8)
        else:
            assert result is not None
            cleaned = extract_prompt(result)
            self._show_rows(cleaned)
            self.notify("Prompt enhanced — edit rows, or press Ctrl+C to copy.", title="Done", timeout=4)
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.disabled = False
        prompt_input.placeholder = INPUT_PLACEHOLDER
        prompt_input.focus()

    # -- Streaming & spinner -------------------------------------------------------

    def _on_token(self, token: str) -> None:
        # Called on the event loop inside the enhancement worker; just buffer,
        # the flush interval batches UI repaints instead of one per token.
        self._pending.append(token)

    def _flush_stream(self) -> None:
        if not (self._generating and self._pending):
            return
        chunk = "".join(self._pending)
        self._pending.clear()
        self._stream_parts.append(chunk)
        output = self.query_one("#output", TextArea)
        if not self._stream_started:
            # First real tokens: wipe the spinner / QnA text so only the
            # generation is on screen from here on.
            self._stream_started = True
            output.text = ""
        # Append incrementally at the document end — replacing the whole text
        # every flush forced a full relayout and made the pane flicker.
        output.insert(chunk, location=output.document.end)
        output.scroll_end(animate=False)

    def _start_spinner(self, message: str) -> None:
        self._loading_message = message
        self._spinner_active = True
        self._render_loading()

    def _tick_spinner(self) -> None:
        if not self._spinner_active:
            return
        self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
        self._render_loading()

    def _render_loading(self) -> None:
        if not self._spinner_active:
            return
        frame = SPINNER_FRAMES[self._spinner_frame]
        self.query_one("#output", TextArea).text = f"{frame} {self._loading_message} {LOADING_HINT}"

    # -- Clipboard ------------------------------------------------------------------

    @on(Button.Pressed, "#copy-button")
    def _on_copy_pressed(self, event: Button.Pressed) -> None:
        self.action_copy_output()

    def action_copy_output(self) -> None:
        if self._in_rows_view:
            text = "\n".join(self._result_rows)  # copies the EDITED rows
        else:
            text = self.query_one("#output", TextArea).text
        if not text.strip():
            self.notify("Nothing to copy yet.", title="Copy", severity="warning")
            return
        # pyperclip shells out to pbcopy; keep it off the UI thread entirely.
        self.run_worker(
            partial(self._copy_worker, text), group="clipboard", exclusive=True, thread=True
        )

    def _copy_worker(self, text: str) -> None:
        try:
            pyperclip.copy(text)
        except Exception as exc:  # pyperclip raises PyperclipException on missing pbcopy/xclip
            self.call_from_thread(
                self.notify, f"Clipboard unavailable: {exc}", title="Copy", severity="error"
            )
        else:
            self.call_from_thread(
                self.notify, "Copied to clipboard.", title="Clipboard", timeout=2
            )

    # -- Shutdown --------------------------------------------------------------------

    async def action_quit(self) -> None:
        """Ctrl+Q: cancel workers, close HTTP resources, then exit cleanly."""
        self.workers.cancel_all()
        try:
            # wait() re-raises WorkerCancelled/WorkerFailed for cancelled work;
            # a half-finished worker must never crash the shutdown path.
            await self.workers.wait_for_complete()
        except Exception:  # noqa: BLE001
            pass
        await self.manager.aclose()
        if STOP_OLLAMA_ON_EXIT:
            self.manager.shutdown_spawned_daemon()
        self.exit()
