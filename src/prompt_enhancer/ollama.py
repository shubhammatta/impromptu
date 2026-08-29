"""Async Ollama lifecycle management.

Owns everything that talks to the Ollama daemon:

* health probing (`GET /api/tags`)
* spawning `ollama serve` as a detached, non-orphaning background process
* readiness polling (500 ms interval, bounded)
* model verification and streaming `/api/pull` with progress reporting
* streaming `/api/generate` with per-token callbacks

The manager never touches the UI; it reports through callables, which keeps it
unit-testable (see tests/test_ollama_manager.py) and separable from Textual.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import (
    ASK_READ_TIMEOUT,
    ENABLE_THINKING,
    GENERATE_PATH,
    GENERATE_READ_TIMEOUT,
    HEALTH_PATH,
    HEALTH_TIMEOUT,
    NUM_CTX,
    OLLAMA_BIN,
    PULL_PATH,
    PULL_READ_TIMEOUT,
    POLL_INTERVAL,
    SERVE_LOG_PATH,
    STARTUP_TIMEOUT,
    STOP_OLLAMA_ON_EXIT,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
)

EventCallback = Callable[[str], None]


class OllamaError(RuntimeError):
    """Raised for any unrecoverable Ollama daemon/model/HTTP failure."""


def _normalize(model: str) -> str:
    """`qwen3.5` and `qwen3.5:latest` refer to the same model."""
    model = model.strip().lower()
    return model if ":" in model else f"{model}:latest"


def _fmt_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{int(num)} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"  # pragma: no cover - unreachable


class OllamaManager:
    def __init__(
        self,
        host: str,
        model: str,
        *,
        poll_interval: float = POLL_INTERVAL,
        startup_timeout: float = STARTUP_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.poll_interval = poll_interval
        self.startup_timeout = startup_timeout
        self._transport = transport  # injectable for tests
        self._client: httpx.AsyncClient | None = None
        self._daemon: subprocess.Popen[bytes] | None = None
        self._owns_daemon = False

    # -- HTTP plumbing ---------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=httpx.Timeout(HEALTH_TIMEOUT, connect=2.0),
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- Daemon lifecycle ------------------------------------------------------

    async def is_alive(self) -> bool:
        try:
            response = await self._ensure_client().get(
                HEALTH_PATH,
                timeout=httpx.Timeout(HEALTH_TIMEOUT, connect=0.5),
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def list_models(self) -> list[str]:
        response = await self._ensure_client().get(
            HEALTH_PATH, timeout=httpx.Timeout(5.0, connect=2.0)
        )
        response.raise_for_status()
        return [entry.get("name", "") for entry in response.json().get("models", [])]

    def spawn_daemon(self) -> None:
        """Start `ollama serve` detached from our process group and terminal.

        `start_new_session=True` puts the daemon in its own POSIX session, so:

        * Ctrl+C / SIGINT aimed at the TUI never reaches it,
        * it survives our exit and is re-parented to launchd (never a zombie),
        * it can never hold our tty open or block the shell on quit.

        stdout/stderr go to a log file so the daemon can never pollute the
        alternate screen or leave partial output behind after exit.
        """
        binary = OLLAMA_BIN or shutil.which("ollama")
        if not binary:
            raise OllamaError(
                "Ollama is not installed or not on PATH. Install it from "
                "https://ollama.com/download (or set PROMPT_ENHANCER_OLLAMA_BIN) "
                "and relaunch."
            )
        log_path = Path(SERVE_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "ab")  # noqa: SIM115 - child keeps its own fd after we close ours
        try:
            self._daemon = subprocess.Popen(
                [binary, "serve"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()
        self._owns_daemon = True

    def shutdown_spawned_daemon(self) -> None:
        """SIGTERM the daemon we spawned. No-op unless explicitly enabled and owned.

        Default behavior is to leave the daemon running: it is a detached system
        service in its own session, so it cannot block our shell or orphan a
        zombie — it simply stays available for other tools, exactly like
        `brew services` would leave it.
        """
        if not (STOP_OLLAMA_ON_EXIT and self._owns_daemon and self._daemon):
            return
        if self._daemon.poll() is not None:
            return
        try:
            # start_new_session made the child a session/group leader, so a
            # group signal reaches `ollama serve` and any descendants.
            os.killpg(self._daemon.pid, signal.SIGTERM)
            self._daemon.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired, PermissionError):
            pass

    async def ensure_ready(
        self,
        on_event: EventCallback,
        on_progress: EventCallback,
    ) -> None:
        """Full startup sequence: probe -> spawn -> poll -> verify/pull model."""
        on_event(f"Checking Ollama daemon at {self.host} …")
        if await self.is_alive():
            on_event("✓ Ollama daemon is already running.")
        else:
            on_event("○ Daemon not responding — spawning `ollama serve` as a detached process …")
            self.spawn_daemon()
            deadline = time.monotonic() + self.startup_timeout
            while True:
                if await self.is_alive():
                    assert self._daemon is not None  # set by spawn_daemon
                    on_event(f"✓ Ollama daemon is up (spawned pid {self._daemon.pid}).")
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OllamaError(
                        f"Ollama did not become responsive within {int(self.startup_timeout)}s. "
                        f"See {SERVE_LOG_PATH} for daemon output."
                    )
                on_progress(f"◌ Waiting for Ollama to come up … {int(remaining) + 1}s left")
                await asyncio.sleep(self.poll_interval)

        on_event(f"Checking for model `{self.model}` …")
        installed = await self.model_installed()
        if not installed:
            on_event(
                f"◌ Model `{self.model}` is not installed — pulling it now "
                "(Q4 quantized; this can take a few minutes on first run) …"
            )
            await self.pull_model(on_progress)
        on_event(f"✓ Model `{self.model}` is ready.")

    async def model_installed(self) -> bool:
        try:
            names = await self.list_models()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not list installed models: {exc}") from exc
        wanted = _normalize(self.model)
        return wanted in {_normalize(name) for name in names if name}

    async def pull_model(self, on_progress: EventCallback) -> None:
        """Stream `/api/pull`, aggregating per-layer byte progress into one line."""
        layers: dict[str, list[int]] = {}  # digest -> [completed, total]
        last_status = ""
        last_emit = 0.0

        def describe(status: str, force: bool) -> None:
            nonlocal last_emit
            total = sum(t for _, t in layers.values())
            done = sum(c for c, _ in layers.values())
            if total:
                message = (
                    f"◌ Pulling {self.model}: {done / total * 100:5.1f}%  "
                    f"({_fmt_bytes(done)} / {_fmt_bytes(total)})"
                )
            else:
                message = f"◌ Pulling {self.model}: {status or 'starting'} …"
            if status and "digest" not in status:
                message += f"  ·  {status}"
            now = time.monotonic()
            if force or status != last_status or now - last_emit >= 0.25:
                last_emit = now
                on_progress(message)

        try:
            async with self._ensure_client().stream(
                "POST",
                PULL_PATH,
                json={"model": self.model, "stream": True},
                timeout=httpx.Timeout(PULL_READ_TIMEOUT, connect=5.0),
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")
                    raise OllamaError(f"Pull failed (HTTP {response.status_code}): {body[:300]}")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in event:
                        raise OllamaError(f"Pull failed: {event['error']}")
                    status = event.get("status", "")
                    if "digest" in event and "total" in event:
                        layers[event["digest"]] = [
                            int(event.get("completed", 0)),
                            int(event["total"]),
                        ]
                    if status == "success":
                        total = sum(t for _, t in layers.values())
                        on_progress(
                            f"✓ Pull complete — {self.model} ({_fmt_bytes(total)} downloaded)"
                        )
                        return
                    describe(status, force=False)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Pull failed: {exc}") from exc

    # -- Generation --------------------------------------------------------------

    async def ask(self, prompt: str, *, system: str | None = None) -> str:
        """Non-streaming completion — used for short, structured exchanges
        (e.g. the clarifying-question round) where progressive output adds nothing."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": ENABLE_THINKING,
            "options": {"temperature": TEMPERATURE, "top_p": TOP_P, "num_ctx": NUM_CTX},
        }
        if system is not None:
            payload["system"] = system
        try:
            response = await self._ensure_client().post(
                GENERATE_PATH,
                json=payload,
                timeout=httpx.Timeout(ASK_READ_TIMEOUT, connect=5.0),
            )
        except httpx.HTTPError as exc:
            raise OllamaError(f"Request failed: {exc}") from exc
        if response.status_code != 200:
            raise OllamaError(
                f"Request failed (HTTP {response.status_code}): {response.text[:300]}"
            )
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise OllamaError("Malformed (non-JSON) response from Ollama.") from exc
        if "error" in data:
            raise OllamaError(str(data["error"]))
        return str(data.get("response", "")).strip()

    async def generate(
        self,
        prompt: str,
        *,
        on_token: Callable[[str], None] | None = None,
        system: str | None = None,
    ) -> str:
        """Stream `/api/generate`, invoking `on_token` per chunk; returns full text.

        `system` selects the comprehensiveness level's system prompt;
        None falls back to the default (level 5) system prompt.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system if system is not None else SYSTEM_PROMPT,
            "stream": True,
            "think": ENABLE_THINKING,
            "options": {"temperature": TEMPERATURE, "top_p": TOP_P, "num_ctx": NUM_CTX},
        }
        parts: list[str] = []
        try:
            async with self._ensure_client().stream(
                "POST",
                GENERATE_PATH,
                json=payload,
                timeout=httpx.Timeout(GENERATE_READ_TIMEOUT, connect=5.0),
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")
                    raise OllamaError(
                        f"Generation failed (HTTP {response.status_code}): {body[:300]}"
                    )
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in event:
                        raise OllamaError(f"Generation failed: {event['error']}")
                    token = event.get("response") or ""
                    if token:
                        parts.append(token)
                        if on_token is not None:
                            on_token(token)
                    if event.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise OllamaError(f"Generation request failed: {exc}") from exc

        text = "".join(parts).strip()
        if not text:
            raise OllamaError("Model returned an empty response.")
        return text
