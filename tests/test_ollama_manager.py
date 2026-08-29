import json
from types import SimpleNamespace

import httpx
import pytest

import prompt_enhancer.ollama as ollama_module
from prompt_enhancer.ollama import OllamaError, OllamaManager

MODEL = "qwen3.5:9b"


def ndjson(events: list[dict]) -> str:
    return "\n".join(json.dumps(event) for event in events)


def make_manager(handler, **kwargs) -> OllamaManager:
    kwargs.setdefault("poll_interval", 0.01)
    kwargs.setdefault("startup_timeout", 2.0)
    return OllamaManager(
        "http://test", MODEL, transport=httpx.MockTransport(handler), **kwargs
    )


# -- is_alive -------------------------------------------------------------------


async def test_is_alive_true_when_tags_ok():
    def handler(request):
        return httpx.Response(200, json={"models": []})

    assert await make_manager(handler).is_alive() is True


async def test_is_alive_false_on_connect_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    assert await make_manager(handler).is_alive() is False


# -- model_installed --------------------------------------------------------------


async def test_model_installed_exact_match():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": MODEL}, {"name": "llama3:8b"}]})

    assert await make_manager(handler).model_installed() is True


async def test_model_installed_normalizes_missing_tag():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:latest"}]})

    manager = make_manager(handler)
    manager.model = "qwen3.5"
    assert await manager.model_installed() is True


async def test_model_missing():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    assert await make_manager(handler).model_installed() is False


# -- ensure_ready -----------------------------------------------------------------


async def test_ensure_ready_spawns_daemon_when_down(monkeypatch):
    calls = {"tags": 0}

    def handler(request):
        calls["tags"] += 1
        if calls["tags"] <= 2:  # daemon "not up yet" for the first two probes
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"models": [{"name": MODEL}]})

    manager = make_manager(handler)
    spawned = []

    def fake_spawn():
        spawned.append(True)
        manager._daemon = SimpleNamespace(pid=4242)
        manager._owns_daemon = True

    monkeypatch.setattr(manager, "spawn_daemon", fake_spawn)

    events: list[str] = []
    progress: list[str] = []
    await manager.ensure_ready(on_event=events.append, on_progress=progress.append)

    assert spawned == [True]
    assert not any("already running" in event for event in events)
    assert any("spawned pid 4242" in event for event in events)
    assert any("left" in line for line in progress)  # readiness polling was surfaced
    assert events[-1].startswith("✓")


async def test_ensure_ready_times_out_when_daemon_never_comes_up(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    manager = make_manager(handler, startup_timeout=0.05, poll_interval=0.01)
    monkeypatch.setattr(manager, "spawn_daemon", lambda: None)

    with pytest.raises(OllamaError, match="did not become responsive"):
        await manager.ensure_ready(on_event=lambda _: None, on_progress=lambda _: None)


async def test_ensure_ready_pulls_missing_model():
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/pull":
            events = [
                {"status": "pulling manifest"},
                {
                    "status": "pulling sha256:a",
                    "digest": "sha256:a",
                    "total": 100,
                    "completed": 40,
                },
                {
                    "status": "pulling sha256:a",
                    "digest": "sha256:a",
                    "total": 100,
                    "completed": 100,
                },
                {"status": "verifying sha256 digest"},
                {"status": "writing manifest"},
                {"status": "success"},
            ]
            return httpx.Response(200, text=ndjson(events))
        return httpx.Response(404, json={"error": "not found"})

    manager = make_manager(handler)
    events: list[str] = []
    progress: list[str] = []
    await manager.ensure_ready(on_event=events.append, on_progress=progress.append)

    assert any("pulling it now" in event for event in events)
    assert any("40.0%" in line for line in progress)
    assert any("Pull complete" in line for line in progress)
    assert events[-1].startswith("✓") and "is ready" in events[-1]


async def test_ensure_ready_surfaces_pull_error():
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/pull":
            return httpx.Response(
                200, text=ndjson([{"error": "pull model manifest: file not found"}])
            )
        return httpx.Response(404, json={"error": "not found"})

    manager = make_manager(handler)
    with pytest.raises(OllamaError, match="manifest"):
        await manager.ensure_ready(on_event=lambda _: None, on_progress=lambda _: None)


# -- ask (non-streaming) ----------------------------------------------------------


async def test_ask_returns_non_streaming_response():
    captured: dict = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"response": '["Q1?", "Q2?"]', "done": True})

    manager = make_manager(handler)
    text = await manager.ask("prompt here", system="be brief")
    assert text == '["Q1?", "Q2?"]'
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["system"] == "be brief"
    assert captured["payload"]["think"] is False  # thinking off by default (slow)


async def test_ask_surfaces_http_error():
    def handler(request):
        return httpx.Response(500, json={"error": "kaboom"})

    with pytest.raises(OllamaError, match="HTTP 500"):
        await make_manager(handler).ask("x")


async def test_ask_surfaces_malformed_json():
    def handler(request):
        return httpx.Response(200, text='{"a": 1}\n{"b": 2}')  # two documents, not JSON

    with pytest.raises(OllamaError, match="Malformed"):
        await make_manager(handler).ask("x")


# -- generate -----------------------------------------------------------------------


async def test_generate_streams_tokens_to_callback():
    def handler(request):
        assert request.url.path == "/api/generate"
        payload = json.loads(request.content)
        assert payload["model"] == MODEL
        assert payload["stream"] is True
        assert payload["think"] is False  # thinking off by default (slow)
        assert "Expert Prompt Engineer" in payload["system"]
        return httpx.Response(
            200,
            text=ndjson(
                [
                    {"response": "Hello ", "done": False},
                    {"response": "world", "done": False},
                    {"response": "", "done": True},
                ]
            ),
        )

    manager = make_manager(handler)
    tokens: list[str] = []
    text = await manager.generate("crude input", on_token=tokens.append)
    assert text == "Hello world"
    assert tokens == ["Hello ", "world"]


async def test_generate_and_ask_honor_think_opt_in(monkeypatch):
    captured: list[dict] = []

    def handler(request):
        captured.append(json.loads(request.content))
        if request.url.path == "/api/generate":
            return httpx.Response(200, text=ndjson([{"response": "ok", "done": True}]))
        return httpx.Response(200, json={"response": "ok", "done": True})

    monkeypatch.setattr(ollama_module, "ENABLE_THINKING", True)
    manager = make_manager(handler)
    await manager.ask("x", system="s")
    await manager.generate("x", system="s")
    assert all(payload["think"] is True for payload in captured)


async def test_generate_surfaces_ndjson_error():
    def handler(request):
        return httpx.Response(200, text=ndjson([{"error": "model requires more memory"}]))

    with pytest.raises(OllamaError, match="more memory"):
        await make_manager(handler).generate("x")


async def test_generate_surfaces_http_error():
    def handler(request):
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(OllamaError, match="HTTP 404"):
        await make_manager(handler).generate("x")


async def test_generate_rejects_empty_response():
    def handler(request):
        return httpx.Response(200, text=ndjson([{"response": "", "done": True}]))

    with pytest.raises(OllamaError, match="empty response"):
        await make_manager(handler).generate("x")
