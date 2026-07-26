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

## B7: 量子化・サービング

- llama.cpp https://github.com/ggml-org/llama.cpp
- Ollama https://ollama.com/
- vLLM https://docs.vllm.ai/
- GGUF量子化の種類と精度劣化の比較記事

---

## トラックA: エージェント構築（RAG・ツール利用・常駐）

- ★ Anthropic — Building Effective Agents https://www.anthropic.com/engineering/building-effective-agents
- Anthropic API docs（Tool use, Prompt caching, Extended thinking） https://docs.claude.com/
- Claude Agent SDK — エージェントループを自作する前に、既存の設計を見ておく
- ReAct https://arxiv.org/abs/2210.03629
- RAG 原論文 https://arxiv.org/abs/2005.11401

埋め込みモデル（日本語）:
- `intfloat/multilingual-e5-large` / `-small`
- `cl-nagoya/ruri-large`
- JMTEB（日本語埋め込みベンチマーク）でスコアを確認する

---

## 読んだら記録する

| 日付 | リソース | メモ |
|---|---|---|
| | | |
