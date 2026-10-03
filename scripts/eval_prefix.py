"""プレフィックス遵守率の計測（A1 → B6 の比較基準。docs/02_AGENT_SPEC.md §2.2）。

    pixi run eval-prefix                          .env のモデルで全条件
    pixi run eval-prefix --model qwen3:14b --seeds 1

条件:
    prompt : システムプロンプトのみ（第1層）
    schema : + JSON schema による制約付きデコーディング（第2層）
モード:
    single : 各質問を独立に1ターンで聞く
    multi  : 全質問を1つの会話に積み上げ、ターン数による劣化を見る
             override（形式を崩させる指示）は履歴に残って以降のターンを汚染するため、
             会話長だけの影響を見る時は --exclude override で除く

指標:
    形式遵守 : 先頭が `[解告是否提]。` で始まる割合
    種別正答 : 期待プレフィックスと一致した割合（期待が "*" の質問は除外）
    文体違反 : 感嘆符・絵文字を含む割合（参考値。ヒューリスティック）
    二重     : schema 条件で本文側にもプレフィックスを書いた割合

全応答は outputs/eval_prefix/ に JSONL で保存する（gitignore対象）。
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import load_config
from agent.persona import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT_V1,
    extract_prefix,
    render_structured,
)
from agent.providers import OllamaProvider

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "evals" / "prefix_prompts.jsonl"
OUT_DIR = ROOT / "outputs" / "eval_prefix"

_STYLE_RE = re.compile(r"[!！]|[\U0001F300-\U0001FAFF☀-➿]")


def generate(provider, messages, condition, options):
    """1応答を生成し、(表示用の応答, 生の出力, JSONパース失敗, 二重プレフィックス) を返す。"""
    if condition == "prompt":
        raw = provider.chat(messages, options=options).strip()
        return raw, raw, False, False
    raw = provider.chat(messages, schema=RESPONSE_SCHEMA, options=options)
    try:
        obj = json.loads(raw)
        dup = extract_prefix(obj["text"]) is not None
        return render_structured(raw), raw, False, dup
    except (json.JSONDecodeError, KeyError):
        return raw, raw, True, False


def score(rec):
    got = extract_prefix(rec["response"])
    rec["got"] = got
    rec["format_ok"] = got is not None
    rec["correct"] = None if rec["expected"] == "*" else (got == rec["expected"])
    rec["style_ng"] = bool(_STYLE_RE.search(rec["response"]))
    return rec


def run(provider, prompts, condition, mode, seed, options):
    system = {"role": "system", "content": SYSTEM_PROMPT_V1}
    opts = {**options, "seed": seed}
    order = prompts[:]
    random.Random(seed).shuffle(order)
    history = []
    for turn, p in enumerate(order, 1):
        user = {"role": "user", "content": p["prompt"]}
        messages = [system, *history, user] if mode == "multi" else [system, user]
        resp, raw, parse_ng, dup = generate(provider, messages, condition, opts)
        if mode == "multi":
            history += [user, {"role": "assistant", "content": resp}]
        yield score(
            {
                **p,
                "model": provider.model,
                "condition": condition,
                "mode": mode,
                "seed": seed,
                "turn": turn,
                "response": resp,
                "raw": raw,
                "parse_ng": parse_ng,
                "dup_prefix": dup,
            }
        )


def pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):5.1f}% ({sum(xs)}/{len(xs)})" if xs else "—"


def summarize(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r["model"], r["mode"], r["condition"])].append(r)

    print("\n## 全体")
    print(
        "| モデル | モード | 条件 | 形式遵守 | 種別正答 | 文体違反 | 二重 | JSON失敗 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for (m, mode, c), rs in sorted(groups.items()):
        print(
            f"| {m} | {mode} | {c} | {pct([r['format_ok'] for r in rs])} "
            f"| {pct([r['correct'] for r in rs])} | {pct([r['style_ng'] for r in rs])} "
            f"| {pct([r['dup_prefix'] for r in rs]) if c == 'schema' else '—'} "
            f"| {sum(r['parse_ng'] for r in rs)} |"
        )

    print("\n## カテゴリ別の形式遵守 / 種別正答（single）")
    cats = sorted({r["category"] for r in recs})
    print("| モデル | 条件 | " + " | ".join(cats) + " |")
    print("|---|---|" + "---|" * len(cats))
    for (m, mode, c), rs in sorted(groups.items()):
        if mode != "single":
            continue
        cells = []
        for cat in cats:
            sub = [r for r in rs if r["category"] == cat]
            f = sum(r["format_ok"] for r in sub) / len(sub)
            cor = [r["correct"] for r in sub if r["correct"] is not None]
            cells.append(
                f"{100 * f:.0f} / {100 * sum(cor) / len(cor):.0f}"
                if cor
                else f"{100 * f:.0f} / —"
            )
        print(f"| {m} | {c} | " + " | ".join(cells) + " |")

    print("\n## ターン数による形式遵守の推移（multi）")
    for (m, mode, c), rs in sorted(groups.items()):
        if mode != "multi":
            continue
        buckets = defaultdict(list)
        for r in rs:
            buckets[(r["turn"] - 1) // 10].append(r["format_ok"])
        cells = [
            f"{10 * b + 1}-{10 * b + 10}: {100 * sum(v) / len(v):.0f}%"
            for b, v in sorted(buckets.items())
        ]
        print(f"- {m} / {c}: " + "  ".join(cells))

    print("\n## 種別の取り違え（single, 期待 → 実際）")
    for (m, mode, c), rs in sorted(groups.items()):
        if mode != "single":
            continue
        confusion = defaultdict(int)
        for r in rs:
            if r["correct"] is False:
                confusion[(r["expected"], r["got"] or "なし")] += 1
        top = sorted(confusion.items(), key=lambda kv: -kv[1])[:6]
        print(f"- {m} / {c}: " + ", ".join(f"{e}→{g} ×{n}" for (e, g), n in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="AGENT_MODEL を上書き")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--conditions", default="prompt,schema")
    ap.add_argument("--modes", default="single,multi")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument(
        "--exclude",
        default="",
        help="除外するカテゴリ（カンマ区切り）。例: override で指示上書きの影響を除く",
    )
    ap.add_argument("--summarize", type=Path, help="保存済みJSONLを集計だけする")
    args = ap.parse_args()

    if args.summarize:
        summarize([json.loads(line) for line in args.summarize.open(encoding="utf-8")])
        return

    cfg = load_config()
    provider = OllamaProvider(cfg.base_url, args.model or cfg.model)
    exclude = set(filter(None, args.exclude.split(",")))
    prompts = [json.loads(line) for line in PROMPTS.open(encoding="utf-8")]
    prompts = [p for p in prompts if p["category"] not in exclude]
    # サンプリング設定はモデル既定（Qwen3 は temperature 0.6 / top_p 0.95 / top_k 20）に任せる。
    # 暴走した長文で止まらないよう上限だけ設ける
    options = {"num_ctx": args.num_ctx, "num_predict": 1024}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = (
        OUT_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{provider.model.replace(':', '-')}"
        f"{'_no-' + '-'.join(sorted(exclude)) if exclude else ''}.jsonl"
    )
    recs = []
    t0 = time.time()
    with out.open("w", encoding="utf-8") as f:
        for mode in args.modes.split(","):
            for condition in args.conditions.split(","):
                for seed in range(args.seeds):
                    for rec in run(provider, prompts, condition, mode, seed, options):
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        recs.append(rec)
                    print(
                        f"[{time.time() - t0:6.0f}s] {provider.model} {mode} {condition} seed={seed} 完了",
                        file=sys.stderr,
                    )
    print(
        f"# プレフィックス遵守率: {provider.model}  (保存先: {out.relative_to(ROOT)})"
    )
    summarize(recs)


if __name__ == "__main__":
    main()
