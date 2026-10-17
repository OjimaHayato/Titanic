"""データの取得・読み込みと，提出ファイルの書き出し．

パスはすべてプロジェクトのルートを基準にした絶対パスで持つので，ノートブックをどこから
実行しても同じ場所を読み書きする．データが手元になければ Kaggle API でダウンロードする．
ダウンロードには Kaggle の API トークン（~/.kaggle/access_token）と，コンペのルールへの
同意が必要．

  Typical usage example:

  train, test = load_data()
  submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
  save_submission(submission, "02_ml_baseline")
"""
from pathlib import Path

import kagglehub
import pandas as pd

# このファイル（src/titanic/data.py）から 2 つ上がプロジェクトのルート．
# ノートブックをどこから実行しても同じ場所を指すように，ルートを基準にした絶対パスで持つ
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

COMPETITION = "titanic"
N_TEST_ROWS = 418


def download_data() -> None:
    """コンペのデータを data/ にダウンロードする．

    data/train.csv がすでにあれば何もしない．

    Raises:
        kagglehub.exceptions.UnauthenticatedError: Kaggle の API トークンが見つからないとき．
    """
    # kagglehub 1.0.2 はダウンロード済みの output_dir を指定するとエラーになるので，
    # train.csv があるときは呼ばない
    if (DATA_DIR / "train.csv").exists():
        return
    kagglehub.competition_download(COMPETITION, output_dir=str(DATA_DIR))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """train.csv と test.csv を読み込む．

    手元になければ，先に download_data() でダウンロードする．

    Returns:
        (train, test) のタプル．train は形が (891, 12) で目的変数 Survived を含み，
        test は形が (418, 11) で Survived を含まない．
    """
    download_data()
    return pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "test.csv")


def save_submission(submission: pd.DataFrame, name: str) -> Path:
    """提出ファイルの形式を確かめて，submissions/<name>.csv に書き出す．

    Args:
        submission: PassengerId と Survived の 2 列からなる，形が (418, 2) の DataFrame．
            Survived は整数の 0 か 1．
        name: 拡張子を除いたファイル名．ノートブックと同じ名前（例: "02_ml_baseline"）にする．

    Returns:
        書き出したファイルのパス．

    Raises:
        ValueError: 列・行数・Survived の型や値が，コンペの Evaluation に書かれた条件を
            満たさないとき．
    """
    if list(submission.columns) != ["PassengerId", "Survived"]:
        raise ValueError(f"列は PassengerId と Survived の 2 つだけにする: {list(submission.columns)}")
    if len(submission) != N_TEST_ROWS:
        raise ValueError(f"行数は {N_TEST_ROWS} にする: {len(submission)}")
    # bool や float のままだと True や 0.0 と書き出されて，提出エラーになる
    if not pd.api.types.is_integer_dtype(submission["Survived"]):
        raise ValueError(f"Survived は整数型にする: {submission['Survived'].dtype}")
    if not submission["Survived"].isin([0, 1]).all():
        raise ValueError("Survived は 0 か 1 にする")

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    path = SUBMISSIONS_DIR / f"{name}.csv"
    # index=False を付けないと，行番号の列が余分に書き出されて提出エラーになる
    submission.to_csv(path, index=False)
    return path
