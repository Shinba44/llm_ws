# 環境構築 / SETUP

新しいマシンでこのリポジトリをcloneしたら、まずこのファイルの手順を実行する。

---

## 1. 前提

| 項目 | 要件 |
|---|---|
| OS | Linux（Ubuntu想定）。WSL2 / macOS でも可 |
| Python | 3.10 以上 |
| パッケージ管理 | [uv](https://docs.astral.sh/uv/) |
| Git | 任意のバージョン |
| GPU（トラックB用） | NVIDIA GPU + ドライバ + CUDA 12.x |

---

## 2. 共通セットアップ

```bash
git clone git@github.com:Shinba44/llm_ws.git
cd llm_ws

# uv のインストール（未導入の場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係の同期（pyproject.toml 作成後に有効）
uv sync

# APIキーの設定
cp .env.example .env
$EDITOR .env
```

### 動作確認

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. GPUマシンでの追加手順

### 3.1 GPU情報の確認

```bash
nvidia-smi
nvcc --version   # CUDA Toolkit（無くてもPyTorchのwheelだけで動くことが多い）
```

### 3.2 PyTorch（CUDA版）のインストール

`uv sync` で入るのはCPU版の可能性があるため、CUDA版を明示する。
CUDAバージョンに合わせてインデックスURLを変える（下記は CUDA 12.4 の例）。

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

確認:

```bash
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("VRAM(GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY
```

> `bf16 supported: True` なら Ampere 世代以降。学習時は fp16 より bf16 を使う。

### 3.3 学習用ライブラリ（B3・B6で必要になったら）

```bash
uv add transformers datasets accelerate
uv add peft trl bitsandbytes   # B6: LoRA/QLoRA
uv add tensorboard             # or wandb
```

---

## 4. マシンプロファイル

**新しいマシンで作業を始めたら、この表に追記してコミットすること。**
「あのマシンではどこまでやれたか」を思い出すために使う。

| マシン名 | CPU | RAM | GPU / VRAM | 用途 | 備考 |
|---|---|---|---|---|---|
| laptop-i5 | Intel i5-7300U (4T) | 7GB | なし | ドキュメント・トラックA開発 | 学習は不可。ローカル推論も0.6B Q4クラスが限界 |
| （GPU機） | | | | トラックB本番 | **← 記入してください** |

記入用コマンド:

```bash
echo "CPU: $(LANG=C lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "RAM: $(free -g | awk '/^Mem:/{print $2}') GB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: none"
```

---

## 5. データとチェックポイントの扱い

`.gitignore` により以下はコミットされない:

- `data/`, `datasets/` — 学習データ
- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.gguf` — モデル重み
- `.env` — APIキー

**マシンをまたぐ時の運用**:

| 対象 | 方法 |
|---|---|
| 学習データ | `scripts/download_data.py` で再取得（作成予定） |
| 学習済み重み | HuggingFace Hub のプライベートリポジトリ経由 |
| 実験ログ | `wandb` or `runs/` を要約して `03_PROGRESS.md` に記録 |
| APIキー | パスワードマネージャから手動で `.env` に設定 |

---

## 6. GPUを持ち出せない時の逃げ道

| 手段 | GPU | 制限 |
|---|---|---|
| Google Colab（無料） | T4 16GB | セッション切断あり。`notebooks/` に置いて共有 |
| Kaggle Notebooks | T4×2 / P100 | 週30時間 |
| Colab Pro | A100/L4 | 有料 |

Colabで作業する場合は、ノートブック冒頭でこのリポジトリをcloneして使う:

```python
!git clone https://github.com/Shinba44/llm_ws.git /content/llm_ws
%cd /content/llm_ws
!pip install -q -e .
```

---

## 7. トラブルシューティング

| 症状 | 対処 |
|---|---|
| `torch.cuda.is_available()` が False | ドライバ確認 → `nvidia-smi`。CPU版torchが入っている可能性 → §3.2 |
| CUDA out of memory | batch size を下げる / 勾配累積 / `torch.cuda.empty_cache()` / bf16化 |
| 日本語が文字化け | ロケール確認。`LANG=ja_JP.UTF-8` |
| uv sync が遅い | `UV_HTTP_TIMEOUT=120` を設定 |
