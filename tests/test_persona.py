import json

import pytest

from agent.persona import RESPONSE_SCHEMA, extract_prefix, render_structured


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("解。当該関数は3箇所から参照されています。", "解"),
        ("是。", "是"),
        ("  否。当該情報は保有していません。", "否"),
        ("提。以下の修正を推奨します。", "提"),
        ("告。テストが1件失敗しています。", "告"),
        ("解 当該関数は…", None),  # 句点が無い
        ("答。…", None),  # 未定義のプレフィックス
        ("はい、解。です", None),  # 先頭でない
        ("", None),
    ],
)
def test_extract_prefix(text, expected):
    assert extract_prefix(text) == expected


def test_render_structured():
    raw = json.dumps(
        {"prefix": "解", "text": "リストは変更可能です。"}, ensure_ascii=False
    )
    assert render_structured(raw) == "解。リストは変更可能です。"


def test_render_structured_drops_duplicated_prefix():
    raw = json.dumps(
        {"prefix": "否", "text": "否。当該情報は保有していません。"}, ensure_ascii=False
    )
    assert render_structured(raw) == "否。当該情報は保有していません。"


def test_schema_enum_matches_prefixes():
    assert RESPONSE_SCHEMA["properties"]["prefix"]["enum"] == [
        "解",
        "告",
        "是",
        "否",
        "提",
    ]
