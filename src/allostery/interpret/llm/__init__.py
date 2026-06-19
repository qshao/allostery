from __future__ import annotations

from typing import Protocol


class LLMBackend(Protocol):
    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        ...


def make_backend(
    name: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMBackend:
    if name == "ollama":
        from allostery.interpret.llm.ollama import OllamaBackend
        return OllamaBackend(
            model=model or "qwen3",
            base_url=base_url or "http://localhost:11434",
        )
    if name == "anthropic":
        from allostery.interpret.llm.anthropic import AnthropicBackend
        return AnthropicBackend(model=model or "claude-opus-4-8")
    if name == "openai":
        from allostery.interpret.llm.openai import OpenAIBackend
        return OpenAIBackend(model=model or "gpt-4.1")
    raise ValueError(f"unknown llm backend {name!r}; expected ollama, anthropic, or openai")


__all__ = ["LLMBackend", "make_backend"]
