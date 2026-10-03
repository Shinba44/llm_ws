"""Ollama のネイティブAPI（/api/chat）を叩くプロバイダ。

OpenAI互換の /v1 ではなくネイティブAPIを使う理由:
- `think: false` で Qwen3 の思考モードを確実に切れる（即答用途なので常に切る）
- `format` に JSON schema を渡して制約付きデコーディングができる
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx

from agent.providers.base import Message


class OllamaProvider:
    def __init__(self, base_url: str, model: str, timeout: float = 600.0):
        # .env は OpenAI互換の URL（…/v1）で書く約束なので、ネイティブAPI用に末尾を落とす
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def _body(self, messages, stream, schema, options) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": False,
        }
        if schema is not None:
            body["format"] = schema
        if options:
            body["options"] = options
        return body

    def chat(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        r = self._client.post(
            f"{self.base_url}/api/chat",
            json=self._body(messages, False, schema, options),
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def stream(
        self,
        messages: list[Message],
        *,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=self._body(messages, True, None, options),
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if piece := chunk.get("message", {}).get("content"):
                    yield piece
                if chunk.get("done"):
                    break
