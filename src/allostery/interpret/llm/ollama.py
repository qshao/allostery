from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable


class OllamaBackend:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._urlopen = urlopen or urllib.request.urlopen

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        return json.loads(body["message"]["content"])


__all__ = ["OllamaBackend"]
