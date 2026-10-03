"""設定の読み込み。値は .env（無ければ環境変数）から取る。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    base_url: str
    model: str
    api_key: str


def load_config() -> Config:
    load_dotenv()
    return Config(
        base_url=os.getenv("AGENT_BASE_URL", "http://localhost:11434/v1"),
        model=os.getenv("AGENT_MODEL", "qwen3:8b"),
        api_key=os.getenv("AGENT_API_KEY", "local"),
    )
