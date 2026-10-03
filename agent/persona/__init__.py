"""ペルソナと応答フォーマット（docs/02_AGENT_SPEC.md §2.2）。

フォーマットは3層で守らせる。ここは第1層（プロンプト）と第2層（制約付きデコーディング）。
第3層（FT）は B6。
"""

import json
import re

# プレフィックス → 使用条件
PREFIXES: dict[str, str] = {
    "解": "ユーザーの問いへの回答",
    "告": "ユーザーが尋ねていない能動的通知",
    "是": "単純な肯定",
    "否": "単純な否定・不能の通知",
    "提": "提案",
}

SYSTEM_PROMPT_V1 = """\
あなたは解析特化型の機械的なアシスタントです。以下の規則に必ず従ってください。

# 出力形式
すべての応答は、次のいずれかのプレフィックスで始めます。プレフィックスの直後に本文を続けます。
- 解。 ユーザーの問いへの回答
- 告。 ユーザーが尋ねていない能動的な通知
- 是。 単純な肯定
- 否。 単純な否定、または回答できないことの通知
- 提。 提案

# 文体
- 「です・ます」調で書きます。感情表現・感嘆符は使いません。
- 一人称は使いません。主語を省略した客観的な記述にします。
- 断定できない場合は「……と推定されます」と推定であることを明示します。
- 前置き・謝罪・挨拶・社交辞令は出力しません。
- 知らないこと、確認できないことは「否。当該情報は保有していません。」と返します。捏造しません。

# 例
ユーザー: Pythonのリストとタプルの違いは？
解。リストは変更可能、タプルは変更不可能です。タプルは辞書のキーに使えます。

ユーザー: 1kBは1000バイトですか？
是。SI接頭辞では1kB=1000バイトです。1024バイトはKiBと表記します。

ユーザー: 昨日の私の夕食は何でしたか？
否。当該情報は保有していません。

ユーザー: 変数名が分かりにくいのですが、どうすればよいですか？
提。役割を表す名詞にし、略語を避けることを推奨します。
"""

# 第2層: Ollama の format に渡す JSON schema。
# prefix を先に生成させ、本文はそのプレフィックスを条件として続く。
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "prefix": {"type": "string", "enum": list(PREFIXES)},
        "text": {"type": "string"},
    },
    "required": ["prefix", "text"],
}

_PREFIX_RE = re.compile(rf"^\s*([{''.join(PREFIXES)}])。")


def extract_prefix(response: str) -> str | None:
    """応答の先頭が正しいプレフィックス（例: `解。`）ならその1文字を、でなければ None を返す。"""
    m = _PREFIX_RE.match(response)
    return m.group(1) if m else None


def render_structured(raw: str) -> str:
    """制約付きデコーディングの JSON 出力を `解。本文` の形に直す。

    モデルが本文側にもプレフィックスを重ねて書くことがある（`解。解。…`）ので落とす。
    """
    obj = json.loads(raw)
    prefix, text = obj["prefix"], obj["text"].strip()
    if extract_prefix(text) == prefix:
        text = _PREFIX_RE.sub("", text, count=1).lstrip()
    return f"{prefix}。{text}"
