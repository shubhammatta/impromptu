"""Shell-style history for submitted prompts.

Keeps the in-session list plus an optional on-disk file (one entry per line,
most recent last) so past prompts survive app restarts. Traversal mirrors a
shell: Up steps to older entries (stashing what the user was typing as a
draft), Down steps forward, and stepping past the newest entry restores the
draft.
"""

from __future__ import annotations

from pathlib import Path


class PromptHistory:
    def __init__(self, path: str | Path | None = None, max_entries: int = 100) -> None:
        self.path = Path(path) if path is not None else None
        self.max_entries = max_entries
        self.entries: list[str] = []  # oldest first, capped at max_entries
        self._browse: int | None = None  # index into entries while traversing
        self._draft = ""  # value being typed before traversal started
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError, ValueError):
            return  # an unreadable history file must never block startup
        self.entries = [line for line in (raw.strip() for raw in lines) if line][
            -self.max_entries :
        ]

    def _persist(self, entry: str) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(entry + "\n")
        except OSError:
            pass

    # -- recording -------------------------------------------------------------

    def add(self, entry: str) -> None:
        entry = entry.strip()
        if not entry:
            return
        self.reset_browse()
        if self.entries and self.entries[-1] == entry:
            return  # no duplicate runs (immediate re-submit)
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            del self.entries[: len(self.entries) - self.max_entries]
        self._persist(entry)

    # -- traversal ---------------------------------------------------------------

    def reset_browse(self) -> None:
        self._browse = None
        self._draft = ""

    def older(self, current: str) -> str | None:
        """Step back through history; `current` is stashed as the draft on entry."""
        if not self.entries:
            return None
        if self._browse is None:
            self._draft = current
            self._browse = len(self.entries) - 1
        elif self._browse > 0:
            self._browse -= 1
        return self.entries[self._browse]

    def newer(self) -> str | None:
        """Step forward; past the newest entry restores the pre-traversal draft."""
        if self._browse is None:
            return None
        if self._browse < len(self.entries) - 1:
            self._browse += 1
            return self.entries[self._browse]
        draft = self._draft
        self.reset_browse()
        return draft
