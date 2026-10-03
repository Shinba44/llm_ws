# CLAUDE.md

Claude Code がこのリポジトリで作業する際の文脈。

## プロジェクト概要

特定個人に常駐し、対象を解析し、能動的に通知してくるパーソナルLLMエージェントを自作する
学習プロジェクト。トラックA（エージェント構築）とトラックB（LLMスクラッチ実装）を並走させる。

商用LLM APIを使わず、推論はすべてローカルGPUで完結させる。その結果、小型モデルでは
プロンプトだけで応答形式を守れないため、**ファインチューニング（B6）が必須工程**になる。
これが2つのトラックの合流点（M3）。

## 最初に読むファイル

1. `docs/03_PROGRESS.md` — 現在の状態と次のアクション。**必ず最初に読む**
2. `docs/00_PLAN.md` — 全体計画。フェーズID（A0, A1, B2 など）はここで定義されている
3. `docs/02_AGENT_SPEC.md` — エージェント仕様。ペルソナ実装はこれを正とする

---

## 現在の状態（2026-07-30 時点）

| 項目 | 値 |
|---|---|
| マイルストーン | **M0 完了** → **M1「ローカルモデルと会話できる」進行中** |
| 進行中フェーズ | **A0: ローカル推論基盤**（Ollama導入と実測） |
| 次の実装 | A1: 対話CLI。コードはまだ1行も無い |

> **粒度はマイルストーン単位まで。** タスク単位の進捗・作業ログ・未決定事項は
> `docs/03_PROGRESS.md` が正。マイルストーンが変わった時だけこの節を更新する。

**完了済み**: pixi環境（3環境）、GPU機のセットアップと動作確認、計画ドキュメント一式、
A5の視覚仕様と参照実装。

**未着手**: `agent/` パッケージ全体、`scratch/`（トラックB）全体。
`pixi run agent` は `agent/cli.py` が無いため現状failする。

---

## マシン

| 名前 | 構成 | 使う環境 |
|---|---|---|
| **roboworks**（Alienware Area-51 R4） | i9-7900X 10C/20T / RAM 62GB / **GTX 1080 Ti ×2** | `gpu` |
| **laptop-i5** | i5-7300U / RAM 7GB / GPUなし | `laptop` |

GPU機の実効VRAMは **GPU0が約9.6GB**（デスクトップ描画で消費）、**GPU1が約10.7GB**。
学習は `CUDA_VISIBLE_DEVICES=1` でGPU1に寄せる。

> **⚠️ 起動時にGPUが1枚認識されないことがある**（2026-09〜10に3回発生。消えるカードは毎回異なる）。
> 1枚欠けると番号が詰まり、`CUDA_VISIBLE_DEVICES=1` ではGPUが見えなくなる
> （`torch.cuda.is_available()` のフォールバックでCPU実行になり、気づきにくい）。
> **作業前に `nvidia-smi -L` で2枚あることを確認する。** 欠けていたら電源ケーブルを抜いて
> 完全に電源を落としてから起動し直すと復帰する。詳細は `docs/01_SETUP.md` §6.3。

---

## ★ハード制約: GPUがPascal世代である

GTX 1080 Ti は **compute capability 6.1（Pascal）**。実機で確認済みの制約。
**これらに反する提案・実装をしない。**

| 禁止事項 | 理由 |
|---|---|
| **torch を 2.8以上に上げない** | 2.8以降の cu128/cu129 は sm_60/sm_61 のカーネルを含まない |
| **wheel index を cu128 / cu130 に変えない** | 同上。`cu126` + `torch<2.8` を `pixi.toml` で固定済み |
| **bf16 を前提にしたコードを書かない** | ネイティブ非対応。学習は **fp32主体**で組む |
| **FlashAttention を使わない** | sm75/sm80以上が必要 |
| **QLoRA（bitsandbytes 4bit）を無検証で前提にしない** | sm75以上想定。動かなければ通常LoRAで1B級に落とす |

補足:
- `nvidia-smi` の `CUDA Version: 13.0` は**ドライバの対応上限**であり、GPUの対応上限ではない。
  index の選択は必ず compute capability で判断する。
- `torch.cuda.get_arch_list()` に `sm_61` は無いが、CUDAの前方互換規則
  （X.y向けcubinはX.z, z≥y で動く）により **`sm_60` のカーネルで動作する**。
- `torch.cuda.is_bf16_supported()` は既定でエミュレーションを含むため Pascal でも `True`
  を返す。判定には `including_emulation=False` を使う。
- 検証は `pixi run -e gpu gpu-check`。詳細は `docs/01_SETUP.md` §6.1〜6.2。

---

## 作業時のルール

- **作業を終えたら `docs/03_PROGRESS.md` の「作業ログ」「次のアクション」「進捗表」を更新する。**
  複数マシンをまたいで作業するため、このファイルが唯一の引き継ぎ手段。
  マイルストーンが変わった時は、このファイルの「現在の状態」も合わせて直す。
- 新しいマシンで作業を始めた場合は `docs/01_SETUP.md` §6 のマシンプロファイル表に追記する。
- **商用LLM APIを使わない。推論はすべてローカル**（Ollama / llama.cpp server）。
  Anthropic/OpenAI等のAPIを前提にした実装や提案をしない。詳細は `docs/00_PLAN.md` §9。
  例外はB6のSFTデータ合成のみ。
- 応答フォーマットは**プロンプトだけで守らせようとしない**。
  制約付きデコーディング（GBNF / JSON schema）とファインチューニングを併用する前提。
- **A1では遵守率を実測して記録する。** B6の前後比較の基準値になるため省略しない。
- トラックBの実装は**ライブラリを使う前に一度自分で書く**方針。安易に
  `transformers` で済ませる提案をしない（B6以降は除く）。
- モデル重み・学習データ・`.env` はコミットしない。
- ドキュメントは日本語。コード内のコメントも日本語でよい。

## 環境

- パッケージ管理は **pixi**（`pixi.toml`）。Python・CUDA・cmake・Node.js を一括管理する
- 実行は `pixi run -e <環境> <タスク>`。環境は `laptop`（GPUなし）と `gpu`（GPU機）
- **依存を追加する時は feature を指定する**。例: `pixi add --pypi --feature train datasets`
  共通featureに入れるとGPUなしのマシンで環境が壊れる
- **conda-forge と PyPI を同じ依存チェーンで混ぜない**。torch系はすべて `--pypi`
- `pixi.lock` はコミットする。`.pixi/` はコミットしない
- `[feature.gpu.system-requirements]` に非推奨警告が出るが**これは既知**。
  代替記法が feature 単位で使えないための据え置き。修正しようとしない
- マシンによってGPUの有無が異なる。GPU前提のコードを書く際は
  `torch.cuda.is_available()` でフォールバックする
- このプロジェクトはPythonだけで完結しない（A5はJS/Node、B7はC++/cmake）

## 定義済みタスク

```
pixi run -e gpu gpu-check     GPU動作確認（arch互換・行列積テスト）
pixi run machine-info         マシンプロファイル記入用の情報
pixi run demo                 A5の解析フィールドのデモをブラウザで開く
pixi run lint / fmt / test    ruff / ruff format / pytest
pixi run agent                エージェントCLI（A1で実装するまでfailする）
```
