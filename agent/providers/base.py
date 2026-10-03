from collections.abc import Iterator
from typing import Any, Protocol, TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class Provider(Protocol):
    model: str

    def chat(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """応答全文を返す。schema を渡すと出力をその JSON schema に制約する。"""
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """応答を断片ごとに返す。"""
        ...
