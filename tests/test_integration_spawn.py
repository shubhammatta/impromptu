"""End-to-end lifecycle test: probe fails -> detached spawn -> poll -> generate.

Uses a fake `ollama` binary (a shell shim that execs a stub HTTP server), so the
real spawn/poll/readiness path runs without the actual Ollama daemon.
"""

import json
import socket
import stat
from pathlib import Path

import pytest

import prompt_enhancer.ollama as ollama_module
from prompt_enhancer.ollama import OllamaManager

MODEL = "qwen3.5:9b"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def fake_daemon(tmp_path: Path, monkeypatch):
    """Install a fake `ollama` binary + serve log location; return its port."""
    port = _free_port()
    server_py = tmp_path / "fake_ollama_server.py"
    server_py.write_text(
        (Path(__file__).parent / "fake_ollama_server.py").read_text()
        .replace("11434", str(port), 1)
    )
    shim = tmp_path / "ollama"
    shim.write_text(f"#!/bin/sh\nif [ \"$1\" = serve ]; then\n  exec {fake_daemon_shim_cmd(server_py)}\nfi\nexit 1\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(ollama_module, "OLLAMA_BIN", str(shim))
    monkeypatch.setattr(ollama_module, "SERVE_LOG_PATH", str(tmp_path / "serve.log"))
    return port


def fake_daemon_shim_cmd(server_py: Path) -> str:
    import sys

    return f"'{sys.executable}' '{server_py}'"


async def test_spawn_poll_generate_against_fake_daemon(fake_daemon, monkeypatch):
    manager = OllamaManager(
        f"http://localhost:{fake_daemon}", MODEL, poll_interval=0.05, startup_timeout=10.0
    )

    try:
        events: list[str] = []
        progress: list[str] = []
        await manager.ensure_ready(on_event=events.append, on_progress=progress.append)

        assert any("not responding" in event for event in events)
        assert any("spawned pid" in event for event in events)
        assert any("left" in line for line in progress)  # polling loop was surfaced
        assert events[-1].startswith("✓") and "is ready" in events[-1]

        tokens: list[str] = []
        text = await manager.generate("crude input", on_token=tokens.append)
        assert "fake-daemon persona" in text
        assert len(tokens) == 2  # streamed, not one blob
    finally:
        # Never leak the detached child, even if an assertion above fails.
        monkeypatch.setattr(ollama_module, "STOP_OLLAMA_ON_EXIT", True)
        manager.shutdown_spawned_daemon()
        assert manager._daemon is not None and manager._daemon.poll() is not None


async def test_shutdown_daemon_is_left_running_by_default(fake_daemon):
    manager = OllamaManager(
        f"http://localhost:{fake_daemon}", MODEL, poll_interval=0.05, startup_timeout=10.0
    )
    try:
        await manager.ensure_ready(on_event=lambda _: None, on_progress=lambda _: None)
        assert manager._daemon is not None
        manager.shutdown_spawned_daemon()  # STOP_OLLAMA_ON_EXIT unset -> no-op
        assert manager._daemon.poll() is None  # still alive
    finally:
        # teardown: reap the detached child for real
        import os
        import signal

        try:
            os.killpg(manager._daemon.pid, signal.SIGTERM)
            manager._daemon.wait(timeout=5)
        except (ProcessLookupError, TimeoutError):
            pass
