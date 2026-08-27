"""Unit tests for the shell-style prompt history."""

from prompt_enhancer.history import PromptHistory


def test_add_and_consecutive_dedupe():
    history = PromptHistory()  # in-memory
    history.add("first")
    history.add("second")
    history.add("second")  # immediate re-submit — not recorded twice
    history.add("first")
    assert history.entries == ["first", "second", "first"]


def test_empty_entries_are_ignored():
    history = PromptHistory()
    history.add("   ")
    history.add("")
    assert history.entries == []


def test_older_newer_traversal_and_draft_restore():
    history = PromptHistory()
    history.add("a")
    history.add("b")
    assert history.older("typing something") == "b"
    assert history.older("b") == "a"
    assert history.older("a") == "a"  # stays at the oldest entry
    assert history.newer() == "b"
    assert history.newer() == "typing something"  # draft restored, browse ended
    assert history.newer() is None  # no longer browsing


def test_add_resets_browse_position():
    history = PromptHistory()
    history.add("a")
    history.add("b")
    assert history.older("") == "b"
    history.add("c")  # submitting ends any traversal
    assert history.newer() is None
    assert history.older("") == "c"


def test_empty_history_returns_none():
    history = PromptHistory()
    assert history.older("x") is None
    assert history.newer() is None


def test_persists_and_reloads(tmp_path):
    path = tmp_path / "history.txt"
    first = PromptHistory(path)
    first.add("alpha")
    first.add("beta")

    second = PromptHistory(path)
    assert second.entries == ["alpha", "beta"]
    assert second.older("") == "beta"


def test_max_entries_caps_loaded_and_added():
    path = None
    history = PromptHistory(path, max_entries=3)
    for i in range(5):
        history.add(f"p{i}")
    assert history.entries == ["p2", "p3", "p4"]


def test_max_entries_caps_on_load(tmp_path):
    path = tmp_path / "history.txt"
    path.write_text("\n".join(f"p{i}" for i in range(10)) + "\n", encoding="utf-8")
    history = PromptHistory(path, max_entries=5)
    assert history.entries == ["p5", "p6", "p7", "p8", "p9"]


def test_missing_or_corrupt_file_is_tolerated(tmp_path):
    history = PromptHistory(tmp_path / "does-not-exist.txt")
    assert history.entries == []
    history.add("still works")
    assert history.entries == ["still works"]

    corrupt = tmp_path / "corrupt.bin"
    corrupt.write_bytes(b"\xff\xfe not utf-8 \x00\x01")
    loaded = PromptHistory(corrupt)
    assert isinstance(loaded.entries, list)
