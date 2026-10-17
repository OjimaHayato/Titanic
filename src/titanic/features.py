"""使う特徴量と，欠損値の補完．

04 で決めた特徴量（NUMERIC，CATEGORICAL）と欠損の埋め方をまとめている．05 以降のノートブックは
これを使う．特徴量や埋め方を変えるときは，新しいノートブックで比べてから書き換える．

欠損の埋め方は次のとおり．
Age は全体の中央値で，test の Fare は 3 等の中央値で埋める．どちらも学習データから計算する値なので，
Pipeline の中で埋める（交差検証の分割ごとに，学習用の行だけを使って計算される）．Age は
titanic.modeling.build_pipeline() の数値の前処理が，Fare は IMPUTE_BY を渡したときに
GroupMedianImputer が埋める．Embarked の欠損（train の 2 人）は C で埋める．データから計算する
値ではなく，運賃と等級を見て決めた値なので，fill_embarked() で交差検証の前に埋めてよい．
Cabin は 8 割近くが欠けているので使わない．

  Typical usage example:

  X = fill_embarked(train)
  pipeline = build_pipeline(model, NUMERIC, CATEGORICAL, impute_by=IMPUTE_BY)
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Pclass", "Sex", "Embarked"]

# 全体の中央値ではなく，グループごとの中央値で埋める列（test で Fare が欠けているのは 3 等の 1 人）
IMPUTE_BY = {"Fare": "Pclass"}

# Embarked が欠けている 2 人は，同じチケットの 1 等の乗客で運賃は 80．
# 1 等の運賃の中央値は C が 78.3，S が 52.0 で C に近いので，C とする
EMBARKED_FILL = "C"


def fill_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Embarked の欠損を EMBARKED_FILL で埋めた DataFrame を返す．

    Args:
        df: train.csv か test.csv を読み込んだ DataFrame．

    Returns:
        Embarked の欠損を埋めた，新しい DataFrame．
    """
    return df.fillna({"Embarked": EMBARKED_FILL})


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """グループごとの中央値で，1 つの列の欠損を埋める．

    fit() で学習データからグループごとの中央値を計算し，transform() でその値を使って埋める．
    学習データにないグループや，グループ内がすべて欠損しているときは，全体の中央値で埋める．
    scikit-learn の Pipeline に入れて使う．

    Attributes:
        column: 欠損を埋める列の名前．
        by: グループ分けに使う列の名前．
        medians_: グループごとの中央値の Series．fit() のあとに使える．
        overall_median_: 全体の中央値．fit() のあとに使える．
    """

    def __init__(self, column: str, by: str):
        self.column = column
        self.by = by

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "GroupMedianImputer":
        """グループごとの中央値と，全体の中央値を計算する．

        Args:
            X: column と by を含む学習データ．
            y: 使わない（Pipeline の決まりに合わせるための引数）．

        Returns:
            自分自身．
        """
        self.medians_ = X.groupby(self.by)[self.column].median()
        self.overall_median_ = X[self.column].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """column の欠損を，その行のグループの中央値で埋めた DataFrame を返す．

        Args:
            X: column と by を含む DataFrame．

        Returns:
            column の欠損を埋めた，新しい DataFrame．
        """
        fill = X[self.by].map(self.medians_).fillna(self.overall_median_)
        return X.assign(**{self.column: X[self.column].fillna(fill)})
