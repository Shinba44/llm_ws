"""対話CLI（A1）。

    pixi run agent                 プロンプトのみ（ストリーミング表示）
    pixi run agent --constrained   JSON schema で形式を制約（生成後にまとめて表示）

コマンド: /reset 履歴を消す / /mode 制約の切替 / /exit 終了
"""

import argparse

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from agent.config import load_config
from agent.persona import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT_V1,
    extract_prefix,
    render_structured,
)
from agent.providers import Message, get_provider


def main() -> None:
    ap = argparse.ArgumentParser(description="ローカルLLMエージェントの対話CLI")
    ap.add_argument(
        "--constrained", action="store_true", help="JSON schema で応答形式を制約する"
    )
    ap.add_argument("--model", help="AGENT_MODEL を上書きする")
    args = ap.parse_args()

    cfg = load_config()
    provider = get_provider(cfg)
    if args.model:
        provider.model = args.model
    constrained = args.constrained

    console = Console()
    session: PromptSession = PromptSession(history=InMemoryHistory())
    system: Message = {"role": "system", "content": SYSTEM_PROMPT_V1}
    history: list[Message] = []

    console.print(
        f"[dim]model={provider.model}  constrained={constrained}  /exit で終了[/dim]"
    )

    while True:
        try:
            user = session.prompt("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history.clear()
            console.print("[dim]履歴を消去[/dim]")
            continue
        if user == "/mode":
            constrained = not constrained
            console.print(f"[dim]constrained={constrained}[/dim]")
            continue

        history.append({"role": "user", "content": user})
        messages = [system, *history]
        try:
            if constrained:
                with console.status("生成中…"):
                    reply = render_structured(
                        provider.chat(messages, schema=RESPONSE_SCHEMA)
                    )
                console.print(reply, markup=False)
            else:
                pieces = []
                for piece in provider.stream(messages):
                    pieces.append(piece)
                    console.print(piece, end="", markup=False, highlight=False)
                console.print()
                reply = "".join(pieces).strip()
        except httpx.HTTPError as e:
            history.pop()
            console.print(f"[red]推論サーバへの接続に失敗: {e}[/red]  ({cfg.base_url})")
            continue

        if extract_prefix(reply) is None:
            console.print("[yellow]（プレフィックス不正）[/yellow]")
        # 履歴には表示した形（`解。本文`）で残す
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
