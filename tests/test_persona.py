import json
from pathlib import Path

import pytest

from agent.persona import (
    PROMPTS,
    RESPONSE_SCHEMA,
    extract_prefix,
    make_schema,
    render_structured,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("解。当該関数は3箇所から参照されています。", "解"),
        ("是。", "是"),
        ("  否。当該情報は保有していません。", "否"),
        ("提。以下の修正を推奨します。", "提"),
        ("告。テストが1件失敗しています。", "告"),
        ("問。対象が特定できません。", "問"),
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
        "問",
    ]


def test_v1_keeps_baseline_prefixes():
    # v1 は基準値（2026-10-03）の条件なので `問` を含めない
    prompt, prefixes = PROMPTS["v1"]
    assert prefixes == ["解", "告", "是", "否", "提"]
    assert "問。" not in prompt
    assert make_schema(prefixes)["properties"]["prefix"]["enum"] == prefixes


def test_v2_prompt_mentions_every_prefix():
    prompt, prefixes = PROMPTS["v2"]
    for p in prefixes:
        assert f"- {p}。" in prompt


def test_v2_fewshot_does_not_leak_evalset():
    # few-shot と評価セットの質問が重なると遵守率が水増しされる
    prompt, _ = PROMPTS["v2"]
    root = Path(__file__).resolve().parents[1] / "evals"
    for name in ("prefix_prompts.jsonl", "prefix_prompts_v2.jsonl"):
        for line in (root / name).open(encoding="utf-8"):
            q = json.loads(line)["prompt"]
            assert f"ユーザー: {q}\n" not in prompt, q


def test_v2_evalset_expects_only_known_prefixes():
    _, prefixes = PROMPTS["v2"]
    path = Path(__file__).resolve().parents[1] / "evals" / "prefix_prompts_v2.jsonl"
    expected = {json.loads(line)["expected"] for line in path.open(encoding="utf-8")}
    assert expected <= {*prefixes, "*"}
    assert "問" in expected
