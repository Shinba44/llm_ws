# 環境構築 / SETUP

新しいマシンでこのリポジトリをcloneしたら、まずこのファイルの手順を実行する。

---

## 1. 前提

| 項目 | 要件 |
|---|---|
| OS | Linux（Ubuntu想定）。WSL2 / macOS でも可 |
| パッケージ管理 | [pixi](https://pixi.sh/) |
| Git | 任意のバージョン |
| GPU（トラックB用） | NVIDIA GPU + **ドライバのみ** |

Python本体・CUDA Toolkit・cmake・Node.js はすべて pixi が入れるので、
**システムに事前インストールする必要はない**。GPUマシンで必要なのは NVIDIA ドライバだけ。

### なぜ pixi か

このプロジェクトは Python だけで完結しない。

| フェーズ | 必要になる非Python依存 |
|---|---|
| A5 常駐オーバーレイ | Node.js（Electron採用時） |
| **B7 量子化・高速化** | **cmake + C++コンパイラ**（llama.cpp のビルド） |
| B3 学習 | CUDA Toolkit |

これらを1つのlockfileで管理でき、かつ **マシンごとの差異を `[environments]` で切り替えられる**
のが pixi を選んだ理由。同じ manifest のまま、GPU機では `-e gpu`、ノートPCでは `-e laptop`
を使う。内部のPyPI解決には uv が使われるため、uv の速度は失われない。

---

## 2. 共通セットアップ

```bash
# pixi のインストール（未導入の場合）
curl -fsSL https://pixi.sh/install.sh | sh
# シェルを開き直すか、表示された PATH 設定を反映する

git clone git@github.com:Shinba44/llm_ws.git
cd llm_ws

# APIキーの設定
cp .env.example .env
$EDITOR .env
```

依存関係は `pixi run` / `pixi shell` の初回実行時に自動で解決・インストールされる。
明示的に入れたい場合のみ `pixi install -e <環境名>` を使う。

---

## 3. 環境の使い分け

`pixi.toml` に3つの環境を定義している。**自分のマシンに合うものを使う。**

| 環境 | 用途 | 含まれるもの |
|---|---|---|
| `laptop` | GPUなしのマシン | エージェント + CPU版torch + Node.js |
| `gpu` | GPU機。トラックB本番 | エージェント + CUDA版torch + 学習ライブラリ + ビルドツール |
| `default` | 最小構成 | エージェント + CPU版torch |

```bash
# GPU機で学習を回す
pixi run -e gpu python scratch/b3_training/train.py

# ノートPCでエージェントを動かす
pixi run -e laptop agent

# シェルに入って作業する
pixi shell -e gpu
```

### 定義済みタスク

```bash
pixi task list          # 一覧
pixi run gpu-check      # GPU情報の確認
pixi run machine-info   # マシンプロファイル記入用の情報を出力
pixi run demo           # 解析フィールドのデモをブラウザで開く
```

---

## 4. GPUマシンでの確認

### 4.1 ドライバの確認

```bash
nvidia-smi
```

これが通れば準備完了。CUDA Toolkit をシステムに入れる必要はない（pixi が管理する）。

### 4.2 PyTorchからGPUが見えるか

```bash
pixi run -e gpu gpu-check
```

期待する出力:

```
torch: 2.x.x+cu124
cuda available: True
device: NVIDIA GeForce RTX ...
VRAM(GB): 12.0
bf16 supported: True
```

> `bf16 supported: True` なら Ampere 世代以降。学習時は fp16 より bf16 を使う。

`cuda available: False` になる場合は §8 を参照。

### 4.3 CUDAバージョンの選択 ★GPU機で最初に確認すること

`pixi.toml` の既定は **CUDA 12.8**（torch 2.11.0）。互換性を優先した設定なので、
**GPU機のドライバが新しければ上げたほうがよい。**

判断材料は `nvidia-smi` の右上に出る `CUDA Version:` の値。これが**ドライバが対応する上限**で、
これを超えるwheelを入れると実行時にドライバエラーになる。

| `nvidia-smi` の CUDA Version | 使うindex | 解決される torch | 備考 |
|---|---|---|---|
| 13.0 以上 | `cu130` | **2.13.0** | CPU環境と同一バージョンになる。最も望ましい |
| 12.8 〜 12.9 | `cu128` | 2.11.0 | **現在の既定** |
| 12.4 〜 12.7 | `cu124` | 2.6.0 | かなり古い。ドライバ更新を検討する |

変更は `pixi.toml` の1箇所だけ:

```toml
[feature.gpu.pypi-dependencies]
torch = { version = ">=2.5", index = "https://download.pytorch.org/whl/cu128" }
```

変更後は `pixi lock` を実行し、**`pixi.lock` も一緒にコミットする**。

> 上表は実際に `pixi lock` を回して確認した結果（2026-07時点）。
> PyTorch側のwheel供給は変わるので、思ったバージョンが降ってこない時は
> `pixi lock` の出力で実際に解決されたバージョンを確認すること。

---

## 5. 依存関係を追加する

**追加先の feature を間違えないこと。** ノートPCで動かないパッケージを共通featureに入れると、
laptop環境が壊れる。

```bash
# トラックAのエージェント用（全環境で使う）
pixi add --pypi --feature agent httpx

# 学習用（gpu環境のみ）
pixi add --pypi --feature train transformers datasets accelerate
pixi add --pypi --feature train peft trl bitsandbytes

# ビルドツール（conda-forge から）
pixi add --feature native cmake ninja
```

> **重要**: conda-forge と PyPI で同じ依存チェーンを混ぜない。
> torch を PyPI から入れているので、**torch に依存するパッケージはすべて `--pypi` で入れる**。
> 混ぜると依存解決が壊れる。

追加後は `pixi.lock` もコミットする。これがマシン間で環境を一致させる根拠になる。

---

## 6. マシンプロファイル

**新しいマシンで作業を始めたら、この表に追記してコミットすること。**
「あのマシンではどこまでやれたか」を思い出すために使う。

| マシン名 | CPU | RAM | GPU / VRAM | 使用環境 | 備考 |
|---|---|---|---|---|---|
| laptop-i5 | Intel i5-7300U (4T) | 7GB | なし | `laptop` | 学習は不可。ローカル推論も0.6B Q4クラスが限界 |
| （GPU機） | | | | `gpu` | **← 記入してください** |

記入用の情報は次のコマンドで出る:

```bash
pixi run machine-info
```

---

## 7. データとチェックポイントの扱い

`.gitignore` により以下はコミットされない:

- `data/`, `datasets/` — 学習データ
- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.gguf` — モデル重み
- `.env` — APIキー
- `.pixi/` — pixi が作る仮想環境の実体

**`pixi.lock` はコミットする。** これがマシン間の再現性を担保する。

**マシンをまたぐ時の運用**:

| 対象 | 方法 |
|---|---|
| 環境 | `pixi.lock` をコミット → 別マシンで `pixi install -e <環境名>` |
| 学習データ | `scripts/download_data.py` で再取得（作成予定） |
| 学習済み重み | HuggingFace Hub のプライベートリポジトリ経由 |
| 実験ログ | `wandb` or `runs/` を要約して `03_PROGRESS.md` に記録 |
| APIキー | パスワードマネージャから手動で `.env` に設定 |

### モデル重みのキャッシュ

モデル重みは環境管理とは別軸で扱う。`HF_HOME` を共有ディレクトリに向けると、
複数の環境・プロジェクトでダウンロード済みの重みを共有できる。

```bash
# .env に書いておく
HF_HOME=/home/<user>/.cache/huggingface
```

---

## 8. GPUを持ち出せない時の逃げ道

| 手段 | GPU | 制限 |
|---|---|---|
| Google Colab（無料） | T4 16GB | セッション切断あり。`notebooks/` に置いて共有 |
| Kaggle Notebooks | T4×2 / P100 | 週30時間 |
| Colab Pro | A100/L4 | 有料 |

ColabにはPython環境が既にあるため pixi は使わない。素の pip で必要なものだけ入れる。

```python
!git clone https://github.com/Shinba44/llm_ws.git /content/llm_ws
%cd /content/llm_ws
!pip install -q transformers datasets accelerate peft trl
```

> Colabのtorchバージョンは `pixi.lock` と一致しない。**Colabは実験用と割り切る。**
> 本番の学習結果は必ずGPU機で再現できることを確認する。

---

## 9. トラブルシューティング

| 症状 | 対処 |
|---|---|
| `torch.cuda.is_available()` が False | ① `nvidia-smi` が通るか ② `-e gpu` を指定したか ③ `[feature.gpu.system-requirements] cuda` を宣言したか（これが無いとCPU版が選ばれる） |
| 依存解決が失敗する | conda-forge と PyPI を混ぜていないか確認。torch系はすべて `--pypi` |
| CUDA out of memory | batch size を下げる / 勾配累積 / bf16化 / `torch.cuda.empty_cache()` |
| lockfileの競合 | `pixi.lock` は再生成できる。競合したら片方を採用して `pixi install` で作り直す |
| 環境が壊れた | `rm -rf .pixi && pixi install -e <環境名>` |
| 日本語が文字化け | ロケール確認。`LANG=ja_JP.UTF-8` |

---

## 10. 補足: なぜ Docker ではないのか

完全な再現性を求めるなら Docker + NVIDIA Container Toolkit、あるいは
NGCコンテナ（`nvcr.io/nvidia/pytorch`）という選択肢もある。CUDA・cuDNN・PyTorch・
最適化ライブラリが検証済みの組み合わせで揃っている。

ただしイメージが十数GBあり、学習目的には重い。**M4（常駐エージェントの配布）に
到達するまでは pixi で十分**と判断している。配布段階で必要になったら再検討する。
