from __future__ import annotations

import json

import pytest

from allostery.interpret.llm import make_backend
from allostery.interpret.llm.ollama import OllamaBackend
from allostery.interpret.llm.anthropic import AnthropicBackend
from allostery.interpret.llm.openai import OpenAIBackend


def test_make_backend_dispatch() -> None:
    assert isinstance(make_backend("ollama", model="qwen3"), OllamaBackend)
    assert isinstance(make_backend("anthropic"), AnthropicBackend)
    assert isinstance(make_backend("openai"), OpenAIBackend)
    with pytest.raises(ValueError):
        make_backend("nope")


def test_ollama_backend_parses_response() -> None:
    class _Resp:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
        def read(self) -> bytes:
            return self._payload
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        body = {"message": {"content": json.dumps({"ok": True})}}
        return _Resp(json.dumps(body).encode("utf-8"))

    backend = OllamaBackend(model="qwen3", urlopen=fake_urlopen)
    result = backend.generate_json("sys", "user", {"type": "object"})
    assert result == {"ok": True}
    assert captured["url"].endswith("/api/chat")


def test_anthropic_backend_parses_response() -> None:
    class _Block:
        type = "text"
        text = json.dumps({"summary": "hi"})

    class _Message:
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs):
            assert kwargs["model"] == "claude-opus-4-8"
            assert kwargs["thinking"] == {"type": "adaptive"}
            return _Message()

    class _Client:
        messages = _Messages()

    backend = AnthropicBackend(client=_Client())
    assert backend.generate_json("sys", "user", {"type": "object"}) == {"summary": "hi"}


def test_openai_backend_parses_response() -> None:
    class _Choice:
        class message:  # noqa: N801
            content = json.dumps({"summary": "hi"})

    class _Completions:
        def create(self, **kwargs):
            class _R:
                choices = [_Choice()]
            return _R()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    backend = OpenAIBackend(client=_Client())
    assert backend.generate_json("sys", "user", {"type": "object"}) == {"summary": "hi"}
