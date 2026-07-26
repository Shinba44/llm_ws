# CLAUDE.md

Claude Code がこのリポジトリで作業する際の文脈。

## プロジェクト概要

特定個人に常駐し、対象を解析し、能動的に通知してくるパーソナルLLMエージェントを自作する
学習プロジェクト。トラックA（エージェント構築）とトラックB（LLMスクラッチ実装）を並走させる。

## 最初に読むファイル

1. `docs/03_PROGRESS.md` — 現在の状態と次のアクション。**必ず最初に読む**
2. `docs/00_PLAN.md` — 全体計画。フェーズID（A1, B2 など）はここで定義されている
3. `docs/02_AGENT_SPEC.md` — エージェント仕様。ペルソナ実装はこれを正とする

## 作業時のルール

- **作業を終えたら `docs/03_PROGRESS.md` の「作業ログ」「次のアクション」「進捗表」を更新する。**
  複数マシンをまたいで作業するため、このファイルが唯一の引き継ぎ手段。
- 新しいマシンで作業を始めた場合は `docs/01_SETUP.md` §4 のマシンプロファイル表に追記する。
- **商用LLM APIを使わない。推論はすべてローカル**（Ollama / llama.cpp server）。
  Anthropic/OpenAI等のAPIを前提にした実装や提案をしない。詳細は `docs/00_PLAN.md` §9。
  例外はB6のSFTデータ合成のみ。
- 応答フォーマットは**プロンプトだけで守らせようとしない**。
  制約付きデコーディング（GBNF / JSON schema）とファインチューニングを併用する前提。
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
- マシンによってGPUの有無が異なる。GPU前提のコードを書く際は
  `torch.cuda.is_available()` でフォールバックする
- このプロジェクトはPythonだけで完結しない（A5はJS/Node、B7はC++/cmake）
