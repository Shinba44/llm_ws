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

> `bf16: True` なら Ampere 世代以降で、学習時は fp16 より bf16 を使う。
> `False` の場合は fp32 主体で組む（§6.1）。

`cuda available: False` になる場合は §9 を参照。
`カーネル(sm_XX) : ❌` が出た場合は §6.2 を参照。

### 4.3 CUDAバージョンの選択 ★GPU機で最初に確認すること

`pixi.toml` の既定は **CUDA 12.8**（torch 2.11.0）。互換性を優先した設定なので、
**GPU機のドライバが新しければ上げたほうがよい。**

判断材料は `nvidia-smi` の右上に出る `CUDA Version:` の値。これが**ドライバが対応する上限**で、
これを超えるwheelを入れると実行時にドライバエラーになる。

**判断基準はドライバではなくGPUの世代（compute capability）。**
`nvidia-smi` の CUDA Version はドライバの対応上限であって、GPUの対応上限ではない。

| GPU世代 | cc | 使うindex | 解決される torch |
|---|---|---|---|
| Pascal（GTX 10xx） | 6.0 / 6.1 | **`cu126` + `<2.8` 固定** | **2.7.1** ← **現在の設定** |
| Volta | 7.0 | `cu126` | 2.7.x |
| Turing 以降（RTX 20xx〜） | 7.5+ | `cu128` | 2.11.0 |
| Blackwell | 10.0 / 12.0 | `cu130` | 2.13.0 |

### なぜ Pascal は cu126 に固定するのか

**PyTorch 2.8 以降、CUDA 12.8 / 12.9 ビルドから sm_60 / sm_61 が削除された。**

| ビルド | 同梱アーキテクチャ |
|---|---|
| `cu126` | 5.0 / 6.0 / **6.1** / 7.0 / 7.5 / 8.0 / 8.6 / 9.0 |
| `cu128` | 7.5 / 8.0 / 8.6 / 9.0 / 10.0 / 12.0（**Pascal なし**） |

cu128 を指定するとインストールは成功するが、**実行時にカーネル未対応で落ちる**。
約8GBのダウンロードが無駄になるため、事前に世代を確認すること。

出典:
- [PyTorch Dev Discuss — Maxwell and Pascal architecture support removed in CUDA 12.8 and 12.9 builds](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)
- [ComfyUI Issue #9929 — torch 2.9 incompatible with Pascal architectures](https://github.com/Comfy-Org/ComfyUI/issues/9929)

`pixi.toml` の `<2.8` は Pascal 対応の最後のバージョンに留めるための固定。
GPUを新しい世代に載せ替えるまで緩めないこと。

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

## 5.5 ローカル推論基盤（A0）

本プロジェクトは商用APIを使わない（[00_PLAN.md](00_PLAN.md) §9）。
推論はGPU機に立てたサーバで行う。

### 5.5.1 Ollama（推奨・最短）

pixi管理外のシステムツールとして入れる。

```bash
curl -fsSL https://ollama.com/install.sh | sh

# VRAMに応じて選ぶ（00_PLAN.md §7-2）
ollama pull qwen3:8b

ollama run qwen3:8b "こんにちは"      # 動作確認
```

OpenAI互換エンドポイントが `http://localhost:11434/v1` に立つ。`.env` に設定する。

```bash
AGENT_BASE_URL=http://localhost:11434/v1
AGENT_MODEL=qwen3:8b
AGENT_API_KEY=local
```

### 5.5.2 llama.cpp（GBNF文法制約を使う場合）

応答フォーマットの強制（[02_AGENT_SPEC.md](02_AGENT_SPEC.md) §2.2）にGBNF文法を使うなら
llama.cpp を直接使う。ビルドには `gpu` 環境の `native` feature が要る。

```bash
pixi shell -e gpu          # cmake / ninja / C++コンパイラが入る
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -B build -S ~/llama.cpp -DGGML_CUDA=ON
cmake --build build -j
```

サーバ起動:

```bash
./build/bin/llama-server -m <model>.gguf --host 0.0.0.0 --port 8080 -ngl 99
```

### 5.5.3 ノートPCから使う

GPU機でサーバを `--host 0.0.0.0` で起動し、ノートPCの `.env` を向ける。

```bash
AGENT_BASE_URL=http://<GPU機のIP>:11434/v1
```

これでノートPC側にモデルを置かずにエージェント開発ができる。

### 5.5.4 記録すること

A0の完了条件として、以下を [03_PROGRESS.md](03_PROGRESS.md) に残す。

- 試したモデルと量子化レベル
- tokens/sec と VRAM使用量
- 日本語の品質の印象（実際に喋らせた所感）

---

## 6. マシンプロファイル

**新しいマシンで作業を始めたら、この表に追記してコミットすること。**
「あのマシンではどこまでやれたか」を思い出すために使う。

| マシン名 | CPU | RAM | GPU / VRAM | 使用環境 | 備考 |
|---|---|---|---|---|---|
| laptop-i5 | Intel i5-7300U (4T) | 7GB | なし | `laptop` | 学習は不可。ローカル推論も0.6B Q4クラスが限界 |
| roboworks-Alienware-Area-51-R4 | 未記入 | 未記入 | **GTX 1080 Ti ×2**（各11GB / 計22GB） | `gpu` | **Pascal世代。§6.1 の制約を必ず読むこと** |

ドライバ 580.173.02（CUDA 13.0対応）。GPU0はデスクトップ描画に約1.1GB使用中のため
実効約10GB、GPU1はほぼ空き。CPU/RAMは `pixi run machine-info` で埋めること。

### 6.1 GTX 1080 Ti（Pascal）固有の制約 ★重要

`nvidia-smi` の `CUDA Version: 13.0` は**ドライバの対応上限**であり、
GPUがCUDA 13で動く意味ではない。Pascal (compute capability 6.1) には次の制約がある。

| 制約 | 影響するフェーズ | 対処 |
|---|---|---|
| **PyTorch 2.8+ の cu128/cu129 が sm_61 を削除** | 全般 | **`cu126` + `torch<2.8` に固定済み**（§4.3） |
| **bf16 が使えない** | B3 学習 | fp32 主体。fp16は使えるがPascalは半精度演算が極端に遅く高速化しない |
| **Tensor Core が無い** | B3 / B6 | AMPの恩恵がほぼ無い。学習は現代のGPUより数倍遅いと想定する |
| **FlashAttention 不可**（sm75/sm80以上が必要） | B2 / B3 | 素のattention実装を使う。スクラッチ実装が目的なので実害は小さい |
| **bitsandbytes 4bit(QLoRA) が怪しい**（sm75+想定） | B6 | 動かなければ通常のLoRAで1B級に落とす |

**推論（A0）への影響はほぼ無い。** llama.cpp / Ollama はPascalを良好にサポートする。

### 6.2 最初に必ず確認すること

`pixi.toml` は既に Pascal 対応の `cu126` / torch 2.7.1 に固定してあるが、
**インストール後に必ず実機で裏を取る**こと。

```bash
pixi run -e gpu gpu-check
```

- `実行可能カーネル: ✅ sm_60`
- `行列積テスト    : ✅ 成功`

**最終的な証拠は行列積テスト。** これが通れば実際に計算できている。

### cubin の前方互換規則

`arch_list` に `sm_61` が無くても問題ない場合がある。CUDAには次の規則がある。

> compute capability **X.y** 向けにビルドされた cubin は、**X.z（z ≥ y）**
> のデバイスで実行できる（メジャーバージョンが同じ場合に限る）

したがって **`sm_60` のカーネルは cc 6.1 の GTX 1080 Ti で動作する**。
`sm_61` の完全一致を要求すると誤検知になる。

実機（torch 2.7.1+cu126）の `arch_list` は
`sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90` で、cc 6.1 に対して `sm_60` が該当する。

参考までに cu128 の `arch_list` は `sm_75` 以上しかないため、
cc 6.1 に対する実行可能カーネルは**ゼロ**になる。これが cu126 に固定している理由。

### bf16 の表示に注意

`torch.cuda.is_bf16_supported()` は**既定でエミュレーションを含めて判定する**ため、
ネイティブ非対応のPascalでも `True` を返す。`gpu-check` は
`including_emulation=False` で問い合わせて実際のハードウェア対応を表示する。
Pascal では `False` が正しい表示。

### インストール前のチェック

約8GBをダウンロードするので、先に確認しておく。

```bash
df -h ~          # 空きが15GB以上あること
ldd --version    # glibc 2.28 以上であること
nvidia-smi       # ドライバが動いていること
```

### 6.3 2枚のGPUの使い分け

GPU0はディスプレイ出力に使われているため、**計算はGPU1に寄せる**とVRAMを丸ごと使える。

```bash
CUDA_VISIBLE_DEVICES=1 pixi run -e gpu python scratch/b3_training/train.py
```

推論では2枚に分割して**合計22GB**として使える（Ollama/llama.cppは自動で分割する）。
Q4量子化なら32B級のモデルも載る計算になる。ただしPCIe経由の通信が入るため
1枚に収まるモデルより遅くなる。まずは8B級を1枚で動かすのが素直。

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
