from __future__ import annotations

import json
from typing import Any


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4.1", *, client: Any | None = None) -> None:
        self.model = model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "the openai backend requires the 'openai' package: pip install openai"
                ) from exc
            self._client = openai.OpenAI()
        return self._client

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "interpretation", "schema": schema},
            },
        )
        return json.loads(response.choices[0].message.content)


__all__ = ["OpenAIBackend"]
