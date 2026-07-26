# llm_ws

特定個人に常駐し、対象を解析し、能動的に通知してくるパーソナルLLMエージェントを自作する。
同時に、LLMの内部構造を手を動かして理解する。

```
解。当該リポジトリは学習用プロジェクトです。
```

## これは何をするプロジェクトか

**2つのトラックを並走させる。**

- **トラックA「作る」** — 既存モデルの上にエージェントを構築する。早く動かす。
- **トラックB「学ぶ」** — トークナイザ・Transformer・学習ループをスクラッチ実装する。

最終ゴールは両者の合流:
**自分でファインチューンしたモデルが、自作エージェント基盤の上で常駐して動く状態。**

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/00_PLAN.md](docs/00_PLAN.md) | マスタープラン（全体像・フェーズ・マイルストーン） |
| [docs/01_SETUP.md](docs/01_SETUP.md) | 環境構築。**cloneしたらまずここ** |
| [docs/02_AGENT_SPEC.md](docs/02_AGENT_SPEC.md) | エージェントの設計仕様 |
| [docs/03_PROGRESS.md](docs/03_PROGRESS.md) | 進捗ログ。**作業再開時はここを読む** |
| [docs/04_RESOURCES.md](docs/04_RESOURCES.md) | 学習リソース |

## 新しいマシンで作業を始める

```bash
git clone git@github.com:Shinba44/llm_ws.git
cd llm_ws
```

1. [docs/01_SETUP.md](docs/01_SETUP.md) の手順で環境を作る
2. [docs/01_SETUP.md](docs/01_SETUP.md) §4 のマシンプロファイル表に自分のマシンを追記する
3. [docs/03_PROGRESS.md](docs/03_PROGRESS.md) の「次のアクション」から再開する

## 現在の状態

**M0: 基盤整備** — 詳細は [docs/03_PROGRESS.md](docs/03_PROGRESS.md) を参照。

## 作業ルール

- 作業を終える時は必ず `docs/03_PROGRESS.md` を更新してコミットする
- モデル重み・学習データ・APIキーはコミットしない（`.gitignore` 済み）
- トラックBは「動いた」で終わらせず、理解したことを書き残して初めて完了
