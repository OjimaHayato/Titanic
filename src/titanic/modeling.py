"""前処理とモデルをつないだ Pipeline の作成と，交差検証による評価．

04 以降のノートブックは，ここで決めた同じ分割（CV）で交差検証するので，ノートブックをまたいで
スコアをそのまま比べられる．

  Typical usage example:

  X = fill_embarked(train)
  pipeline = build_pipeline(RandomForestClassifier(), NUMERIC, CATEGORICAL, impute_by=IMPUTE_BY)
  scores = {"ランダムフォレスト": cv_accuracy(pipeline, X, train["Survived"])}
  print(summarize(scores))
"""
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic.features import GroupMedianImputer

# 5 分割交差検証を，分け方を変えながら 5 回繰り返す（計 25 回）．1 回だけの 5 分割だと，
# どう分けたかの運で平均が 0.01 程度ぶれるので，繰り返して平均を安定させる
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)


def build_pipeline(
    model: ClassifierMixin,
    numeric: list[str],
    categorical: list[str],
    impute_by: dict[str, str] | None = None,
) -> Pipeline:
    """欠損の補完・前処理・モデルをつないだ Pipeline を作る．

    numeric と categorical に入れなかった列は使わない（ColumnTransformer で落とす）．

    Args:
        model: scikit-learn の分類器．
        numeric: 数値として使う列．残った欠損は中央値で埋め，標準化する．
        categorical: カテゴリとして使う列．欠損は最頻値で埋め，one-hot にする．
        impute_by: 列名からグループの列名への辞書．指定した列は，全体の中央値で埋める前に，
            グループごとの中央値で埋める（例: {"Age": "Title"}）．

    Returns:
        X（add_features() を通した DataFrame）を受け取る Pipeline．
    """
    imputers = [GroupMedianImputer(column, by) for column, by in (impute_by or {}).items()]
    preprocess = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
        ],
        verbose_feature_names_out=False,
    )
    return make_pipeline(*imputers, preprocess, model)


def cv_accuracy(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """CV の分割で交差検証し，各回の正解率を返す．

    Args:
        pipeline: build_pipeline() で作った Pipeline．
        X: 特徴量の DataFrame．
        y: 目的変数（Survived）．

    Returns:
        形が (25,) の，各回の検証用データでの正解率．
    """
    return cross_val_score(pipeline, X, y, cv=CV, scoring="accuracy", n_jobs=-1)


def summarize(scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """交差検証の結果を，名前ごとの平均と標準偏差の表にまとめる．

    Args:
        scores: 名前から，cv_accuracy() が返した正解率の配列への辞書．

    Returns:
        行が名前，列が「平均」と「標準偏差」の DataFrame（小数第 3 位まで）．
    """
    return pd.DataFrame(
        {name: [s.mean(), s.std()] for name, s in scores.items()}, index=["平均", "標準偏差"]
    ).T.round(3)
