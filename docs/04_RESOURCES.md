# 学習リソース

フェーズ別に、実際に使う順で並べている。全部やる必要はない。★は優先。

---

## トラックB全体の導線（最重要）

★ **Andrej Karpathy — Neural Networks: Zero to Hero**
https://karpathy.ai/zero-to-hero.html
このプロジェクトのトラックBは、実質このシリーズをなぞって発展させるもの。
- `micrograd` — 自動微分を最小実装
- `makemore` — 言語モデルの基礎、MLP、BatchNorm
- ★ `Let's build GPT: from scratch` — **B2の教科書**
- ★ `Let's build the GPT Tokenizer` — **B1の教科書**

★ **karpathy/nanoGPT** https://github.com/karpathy/nanoGPT — B3の参照実装
★ **karpathy/minbpe** https://github.com/karpathy/minbpe — B1の参照実装
**karpathy/nanochat** https://github.com/karpathy/nanochat — 事前学習からチャットまで一気通貫の最小実装

---

## B1: トークナイザ

- Karpathy `Let's build the GPT Tokenizer`（動画）
- HuggingFace Tokenizers ドキュメント https://huggingface.co/docs/tokenizers
- SentencePiece 論文（日本語トークナイズを考えるなら）

日本語特有の論点: バイト単位BPE、語彙サイズと圧縮率、トークン効率が英語より悪い問題。

---

## B2: Transformer

- ★ Attention Is All You Need (2017) https://arxiv.org/abs/1706.03762
- ★ The Illustrated Transformer https://jalammar.github.io/illustrated-transformer/ — 図解。最初に読む
- The Annotated Transformer http://nlp.seas.harvard.edu/annotated-transformer/ — 論文をコードで注釈
- RoFormer / RoPE https://arxiv.org/abs/2104.09864
- 書籍『ゼロから作るDeep Learning ❺ 生成モデル編』/『❷ 自然言語処理編』（オライリー）

---

## B3: 学習

- ★ nanoGPT の `train.py` を読む
- Chinchilla: Training Compute-Optimal LLMs https://arxiv.org/abs/2203.15556
- HuggingFace NLP Course https://huggingface.co/learn/nlp-course
- 日本語コーパス: 青空文庫、Wikipedia日本語版ダンプ、[llm-jp-corpus](https://gitlab.llm-jp.nii.ac.jp/)

---

## B4: 推論

- KVキャッシュの解説（nanoGPT / llama.cpp の実装を読むのが早い）
- サンプリング手法: The Curious Case of Neural Text Degeneration (top-p の論文) https://arxiv.org/abs/1904.09751
- Speculative Decoding https://arxiv.org/abs/2211.17192

---

## B6: ファインチューニング

- ★ LoRA https://arxiv.org/abs/2106.09685
- QLoRA https://arxiv.org/abs/2305.14314
- HuggingFace PEFT https://huggingface.co/docs/peft
- ★ TRL (SFTTrainer / DPOTrainer) https://huggingface.co/docs/trl

候補ベースモデル:
- Qwen3 0.6B / 1.7B — 日本語もそこそこ、小さい
- Llama 3.2 1B / 3B
- Gemma 3 系
- 日本語特化: [llm-jp](https://huggingface.co/llm-jp), [Swallow](https://huggingface.co/tokyotech-llm), [Sarashina](https://huggingface.co/sbintuitions)

---

## A0 / B7: ローカル推論・量子化・サービング

**本プロジェクトの土台。** 商用APIを使わないため最初に着手する（00_PLAN.md §9）。

- ★ Ollama https://ollama.com/ — 最短で立つ。OpenAI互換エンドポイント付き
- ★ llama.cpp https://github.com/ggml-org/llama.cpp — GBNF文法制約を使うならこちら
- llama.cpp の server ドキュメント（`examples/server`）
- vLLM https://docs.vllm.ai/ — スループット重視の比較対象
- GGUF量子化の種類（Q4_K_M / Q6_K / Q8_0 / imatrix）と精度劣化の比較記事

## 制約付きデコーディング（A1の要）

小型モデルに出力形式を守らせる技術。APIを使っていたら学ばずに済んだ領域。

- ★ GBNF Guide（llama.cpp `grammars/README.md`）— BNF風の文法で出力を制限する
- Outlines https://github.com/dottxt-ai/outlines — 正規表現/JSON schemaで生成を制約
- Ollama の structured outputs（`format` パラメータ）
- 論文: Guiding LLMs The Right Way / Grammar-Constrained Decoding

## ローカルモデル選定

- Qwen3 https://huggingface.co/Qwen — 日本語も比較的良く、サイズの選択肢が広い
- Gemma 3 https://huggingface.co/google
- 日本語特化: [Sarashina](https://huggingface.co/sbintuitions), [Swallow](https://huggingface.co/tokyotech-llm), [llm-jp](https://huggingface.co/llm-jp)
- [Nejumi LLMリーダーボード](https://wandb.ai/llm-leaderboard) — 日本語性能の比較

---

## トラックA: エージェント構築（RAG・ツール利用・常駐）

APIは使わないが、**エージェントの設計論はプロバイダに依存しない**ので参考にする。

- ★ Anthropic — Building Effective Agents https://www.anthropic.com/engineering/building-effective-agents
  （ワークフローとエージェントの区別、設計パターン。実装非依存で読める）
- ★ ReAct https://arxiv.org/abs/2210.03629 — A4でツール対応モデルが使えない時の代替手段
- RAG 原論文 https://arxiv.org/abs/2005.11401
- Toolformer https://arxiv.org/abs/2302.04761 — ツール利用を学習させる発想（B6の拡張案）

埋め込みモデル（日本語）:
- `intfloat/multilingual-e5-large` / `-small`
- `cl-nagoya/ruri-large`
- JMTEB（日本語埋め込みベンチマーク）でスコアを確認する

---

## 読んだら記録する

| 日付 | リソース | メモ |
|---|---|---|
| | | |
