"""推論バックエンドの抽象化。

いまは Ollama のみ。GBNF文法を使う llama.cpp server や、B6でFTしたモデルは
同じ Provider インターフェースで差し替える。
"""

from agent.config import Config
from agent.providers.base import Message, Provider
from agent.providers.ollama import OllamaProvider

__all__ = ["Message", "OllamaProvider", "Provider", "get_provider"]


def get_provider(cfg: Config) -> Provider:
    return OllamaProvider(base_url=cfg.base_url, model=cfg.model)
