Kaggle Titanic の練習
=====================

## Description

Kaggle の入門コンペ「[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)」で，データの確認から提出までの流れを一通り練習している．乗客の情報から生存（1）か死亡（0）かを当てる二値分類で，評価指標は正解率（Accuracy）．

```
.
├── notebooks/        # 1 実験 = 1 ノートブック．番号付きで増やし，前のものは書き換えない
├── src/titanic/      # 複数のノートブックで使う処理（data.py: データと提出ファイル，visualize.py: グラフ）
├── data/             # Kaggle のデータ．ノートブックを実行すると自動でダウンロードされる
└── submissions/      # 提出ファイル．<ノートブック名>.csv で書き出す
```

`data/` と `submissions/` は git に入れていない．

### 実験の記録

| ノートブック | 内容 | CV | LB |
|---|---|---|---|
| 01_rule_baseline | 性別だけで予測（女性は生存，男性は死亡） | 0.787 | 未提出 |
| 02_ml_baseline | 元の 7 列を使ったランダムフォレスト | 0.825 | 約 0.78 |
| 04_missing_values | 欠損の埋め方を決めた（02 と同じランダムフォレストで評価） | 0.821※ | 提出なし |
| 05_model_selection | 勾配ブースティング（既定値） | 0.820※ | 提出なし |
| 06_tuning | 勾配ブースティング（Optuna で学習率・木の本数・正則化を調整） | 0.833※（探索の最高） | 0.76076 |

CV は 5 分割交差検証（StratifiedKFold，`random_state=0`）での正解率の平均．※ を付けた 04 以降は，5 分割交差検証を 5 回繰り返した（計 25 回）平均．LB は Kaggle に提出したときのスコア．

## Requirement

- Python 3.11 以上
- pandas，scikit-learn，matplotlib，kagglehub，ipykernel（バージョンは `pyproject.toml` に書いてある）
- データのダウンロードに，Kaggle の API トークン（`~/.kaggle/access_token` に置く）と，コンペのページでの Join Competition（ルールへの同意）が要る

## Usage

`notebooks/` のノートブックを番号順に，上から実行する．データが `data/` になければ，最初のセルで自動的にダウンロードされる．

提出ファイルは `submissions/<ノートブック名>.csv` に書き出される．コンペのページの Submit Predictions からアップロードして提出する．

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

`-e` を付けて入れているので，`src/titanic/` を編集した内容は入れ直さなくても反映される．
