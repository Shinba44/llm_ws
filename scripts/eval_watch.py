"""eval_prefix.py の進捗と途中結果をリアルタイム表示する。

    pixi run eval-watch            別ターミナルで起動。Ctrl+C で終了
    pixi run eval-watch --seeds 1  計測時の --seeds に合わせる（総数の計算に使う）

outputs/eval_prefix/ のモデルごとに最新の JSONL を数秒おきに読み直して描き直す。
計測側には手を入れないので、実行中の計測にもそのまま使える。
"""

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "eval_prefix"
N_PROMPTS = sum(
    1 for _ in (ROOT / "evals" / "prefix_prompts.jsonl").open(encoding="utf-8")
)
BLOCKS = [(m, c) for m in ("single", "multi") for c in ("prompt", "schema")]
SPARK = " ▁▂▃▄▅▆▇█"


def load_latest() -> dict[str, tuple[Path, list[dict]]]:
    """モデルごとに最新の JSONL を読む。書きかけの最終行は捨てる。"""
    latest: dict[str, Path] = {}
    for f in sorted(OUT_DIR.glob("*.jsonl")):
        model = f.stem.split("_", 2)[-1]
        latest[model] = f
    out = {}
    for model, f in latest.items():
        recs = []
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if recs:
            out[recs[0]["model"]] = (f, recs)
    return out


def rate(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct_cell(x: float | None) -> Text:
    if x is None:
        return Text("—", style="dim")
    style = "green" if x >= 0.9 else "yellow" if x >= 0.7 else "red"
    return Text(f"{100 * x:5.1f}%", style=style)


def model_panel(model: str, path: Path, recs: list[dict], per_block: int) -> Panel:
    total = per_block * len(BLOCKS)
    done = len(recs)
    age = time.time() - path.stat().st_mtime
    running = done < total and age < 120

    # 速度と残り時間（直近の更新間隔から推定）
    # 開始時刻はファイル名（eval_prefix.py が付ける YYYYmmdd_HHMMSS）から取る。
    # Linux の st_ctime は追記のたびに更新されるので使えない
    start = time.mktime(time.strptime(path.stem[:15], "%Y%m%d_%H%M%S"))
    elapsed = path.stat().st_mtime - start
    eta = ""
    if running and done > 1 and elapsed > 0:
        remain = (total - done) * elapsed / done
        eta = f"  残り約{remain / 60:.0f}分（{time.strftime('%H:%M', time.localtime(time.time() + remain))}ごろ）"
    status = (
        "[green]完了[/]"
        if done >= total
        else "[yellow]計測中[/]"
        if running
        else "[red]停止[/]"
    )

    g = defaultdict(list)
    for r in recs:
        g[(r["mode"], r["condition"])].append(r)

    tbl = Table(box=None, padding=(0, 1), expand=True)
    for col in (
        "",
        "モード",
        "条件",
        "進捗",
        "",
        "形式遵守",
        "種別正答",
        "ターン推移（形式・8ターン毎）",
    ):
        tbl.add_column(col, no_wrap=True)
    for b in BLOCKS:
        xs = g.get(b, [])
        n = len(xs)
        mark = "✅" if n >= per_block else "⏳" if n else "··"
        spark = ""
        if b[0] == "multi" and xs:
            bk = defaultdict(list)
            for r in xs:
                bk[(r["turn"] - 1) // 8].append(r["format_ok"])
            spark = "".join(
                SPARK[round(8 * sum(v) / len(v))] for _, v in sorted(bk.items())
            )
            spark += "  " + " ".join(
                f"{100 * sum(v) / len(v):.0f}" for _, v in sorted(bk.items())
            )
        tbl.add_row(
            mark,
            b[0],
            b[1],
            ProgressBar(total=per_block, completed=n, width=20),
            f"{n}/{per_block}",
            pct_cell(rate(r["format_ok"] for r in xs)),
            pct_cell(rate(r["correct"] for r in xs)),
            spark,
        )

    last = recs[-1]
    last_line = Text.assemble(
        ("直近: ", "dim"),
        (
            f"[{last['mode']}/{last['condition']} seed={last['seed']} turn={last['turn']}] ",
            "dim",
        ),
        (last["prompt"][:30], "cyan"),
        " → ",
        (
            last["response"][:60].replace("\n", " "),
            "green" if last["format_ok"] else "red",
        ),
    )
    head = Text.from_markup(
        f"{status}  {done}/{total} ({100 * done / total:.0f}%){eta}"
    )
    return Panel(
        Group(head, ProgressBar(total=total, completed=done), tbl, last_line),
        title=f"[bold]{model}[/]",
        subtitle=path.name,
    )


def gpu_line() -> Text:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return Text("GPU: 取得できません", style="dim")
    parts = []
    # 脱落したGPU（Xid 79 など）は stdout に出ず stderr にエラーが出る。見落とさないよう赤で出す
    for line in out.stderr.strip().splitlines():
        parts.append(Text(f"⚠ {line}", style="bold red"))
    for line in out.stdout.strip().splitlines():
        i, util, used, tot, temp = (s.strip() for s in line.split(","))
        parts.append(
            Text(
                f"GPU{i} {util:>3}% {int(used) / 1024:4.1f}/{int(tot) / 1024:.1f}GB {temp}℃",
                style="dim",
            )
        )
    return Text("  │  ").join(parts)


def render(per_block: int) -> Group:
    data = load_latest()
    panels = [model_panel(m, f, rs, per_block) for m, (f, rs) in data.items()]
    if not panels:
        panels = [
            Text(
                f"{OUT_DIR} に結果がありません。pixi run eval-prefix を実行してください。"
            )
        ]
    footer = Text(f"更新 {time.strftime('%H:%M:%S')}  Ctrl+C で終了", style="dim")
    return Group(*panels, gpu_line(), footer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds", type=int, default=3, help="計測時の --seeds（総数の計算に使う）"
    )
    ap.add_argument("--interval", type=float, default=2.0, help="更新間隔（秒）")
    args = ap.parse_args()
    per_block = N_PROMPTS * args.seeds

    with Live(render(per_block), refresh_per_second=4, screen=False) as live:
        try:
            while True:
                time.sleep(args.interval)
                live.update(render(per_block))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
