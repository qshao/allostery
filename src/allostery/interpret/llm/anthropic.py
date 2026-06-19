from __future__ import annotations

import json
from typing import Any


class AnthropicBackend:
    def __init__(self, model: str = "claude-opus-4-8", *, client: Any | None = None) -> None:
        self.model = model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "the anthropic backend requires the 'anthropic' package: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        client = self._ensure_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next(block.text for block in message.content if block.type == "text")
        return json.loads(text)


__all__ = ["AnthropicBackend"]
