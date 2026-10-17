"""EDA（探索的データ分析）用のグラフを描く関数．

色は役割ごとに固定していて，どのグラフでも生存は青，死亡はオレンジで描く．どの関数も
描いた Axes（グラフ 1 枚分の描画領域）を返すので，plt.subplots() で並べたいときは ax に渡す．
見た目の設定は set_style() を呼んだときだけ変わり，import しただけでは変わらない．

  Typical usage example:

  set_style()
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
  plot_survival_rate(train, "Sex", ax=axes[0])
  plot_survival_rate(train, "Pclass", ax=axes[1])
  plt.show()
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import PercentFormatter

# 色は役割ごとに決めておき，関数の中では役割の名前で使う
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# 生存は青，死亡はオレンジ．どのグラフでも同じ対応にする（色覚の多様性を考慮した配色で検証済み）
SURVIVED_COLOR = "#2a78d6"
DIED_COLOR = "#eb6834"

# 生存率の濃淡（0% が薄く，100% が濃い）．1 つの色相の明るさだけを変える
SURVIVAL_CMAP = LinearSegmentedColormap.from_list(
    "survival",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

# 上から順に探して，最初に見つかった日本語フォントを使う（macOS，Windows，Linux の順）
JAPANESE_FONTS = ["Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans CJK JP", "IPAexGothic"]

# 棒の太さの上限（インチ）．カテゴリが少なくても太い塊にならないようにする
MAX_BAR_INCHES = 0.25


def set_style() -> None:
    """グラフ全体の見た目（日本語フォント，背景，目盛り線など）を設定する．

    matplotlib の rcParams（グラフの既定の見た目を決める設定）を書き換えるので，ノートブックの
    最初に一度だけ呼ぶ．日本語フォントは JAPANESE_FONTS の順に探し，最初に見つかったものを使う．
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    japanese = [name for name in JAPANESE_FONTS if name in available]
    if not japanese:
        print("日本語フォントが見つからないので，日本語が豆腐（□）になるかもしれない")

    plt.rcParams.update({
        "font.family": [*japanese[:1], "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.constrained_layout.use": True,
        "text.color": INK_PRIMARY,
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.titlecolor": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 10,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        # 枠線は下だけ残し，横の目盛り線を細く薄く引く
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 1,
        "ytick.left": False,
        "legend.frameon": False,
    })


def plot_survival_rate(df: pd.DataFrame, column: str, ax: Axes | None = None) -> Axes:
    """column の値ごとの生存率を棒グラフで描く．

    棒の先に生存率を，x 軸に値と人数を書き，全体の生存率を横線で示す．年齢のような連続値は，
    pd.cut で区間に分けた列を作ってから渡す．

    Args:
        df: Survived と column を含む DataFrame．
        column: 値ごとに生存率を出す列の名前．
        ax: 描く先の Axes．省略すると新しい図を作る．

    Returns:
        描いた Axes．
    """
    ax = ax or plt.subplots(figsize=(4, 3.5))[1]
    stats = df.groupby(column, observed=True)["Survived"].agg(["mean", "count"])
    x = np.arange(len(stats))

    ax.bar(x, stats["mean"], width=_bar_width(ax, len(stats)), color=SURVIVED_COLOR)
    for xi, rate in zip(x, stats["mean"]):
        ax.text(xi, rate + 0.02, f"{rate:.0%}", ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)

    overall = df["Survived"].mean()
    ax.axhline(overall, color=INK_MUTED, linewidth=1)

    # 列名はタイトルに入っているので，x 軸には値と人数だけを書く
    ax.set_xticks(x, [f"{value}\nn={n}" for value, n in zip(stats.index, stats["count"])])
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title(f"{column} 別の生存率（横線: 全体 {overall:.0%}）")
    return ax


def plot_survival_heatmap(df: pd.DataFrame, row: str, col: str, ax: Axes | None = None) -> Axes:
    """2 つの列の組み合わせごとの生存率を，色の濃さで描く．

    各マスに生存率と人数を書く．該当する人がいない組み合わせのマスは空白になる．

    Args:
        df: Survived，row，col を含む DataFrame．
        row: 縦に並べる列の名前．
        col: 横に並べる列の名前．
        ax: 描く先の Axes．省略すると新しい図を作る．

    Returns:
        描いた Axes．
    """
    ax = ax or plt.subplots(figsize=(5, 3))[1]
    grouped = df.groupby([row, col], observed=True)["Survived"]
    rates = grouped.mean().unstack()
    counts = grouped.count().unstack()

    ax.imshow(rates.to_numpy(), cmap=SURVIVAL_CMAP, vmin=0, vmax=1, aspect="auto")
    for (i, j), rate in np.ndenumerate(rates.to_numpy()):
        if np.isnan(rate):
            continue
        ax.text(
            j, i, f"{rate:.0%}\nn={counts.iat[i, j]}",
            ha="center", va="center", fontsize=10,
            color=_text_color_on(SURVIVAL_CMAP(rate)),
        )

    ax.set_xticks(range(len(rates.columns)), rates.columns)
    ax.set_yticks(range(len(rates.index)), rates.index)
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    ax.set_title(f"{row} × {col} の生存率")
    # マスの間に背景色の細い隙間を入れ，目盛り線と枠線は消す
    ax.set_xticks(np.arange(len(rates.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rates.index) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)
    return ax


def plot_hist_by_survival(
    df: pd.DataFrame, column: str, bins: int | np.ndarray = 20, ax: Axes | None = None
) -> Axes:
    """column の分布を，生存した人と死亡した人に分けてヒストグラムで重ねて描く．

    column が欠けている人は除き，除いた人数をタイトルに書く．

    Args:
        df: Survived と column を含む DataFrame．
        column: 分布を描く数値の列の名前．
        bins: 区間の数か，形が (区間の数 + 1,) の区間の境界の配列．境界の配列を渡すと，
            範囲外の値は描かない（運賃のように一部だけ極端に大きい値があるときに使う）．
        ax: 描く先の Axes．省略すると新しい図を作る．

    Returns:
        描いた Axes．
    """
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    edges = np.histogram_bin_edges(df[column].dropna(), bins=bins)

    for label, survived, color in [("生存", 1, SURVIVED_COLOR), ("死亡", 0, DIED_COLOR)]:
        values = df.loc[df["Survived"] == survived, column].dropna()
        # 線で輪郭を描き，中は薄く塗る（重なった部分も両方見えるように）
        ax.hist(values, bins=edges, histtype="stepfilled", color=color, alpha=0.10)
        ax.hist(values, bins=edges, histtype="step", color=color, linewidth=2, label=label)

    ax.legend()
    ax.set_xlabel(column)
    ax.set_ylabel("人数")
    n_missing = df[column].isna().sum()
    note = f"，欠損 {n_missing} 人は除く" if n_missing else ""
    ax.set_title(f"{column} の分布（生存・死亡別{note}）")
    return ax


def _bar_width(ax: Axes, n_bars: int) -> float:
    """棒の太さが MAX_BAR_INCHES を超えないように，データ座標での棒の幅を返す．

    Args:
        ax: 棒を描く Axes．
        n_bars: 並べる棒の数．

    Returns:
        ax.bar() の width に渡す幅．大きくても 0.6．
    """
    axes_inches = ax.get_window_extent().width / ax.figure.dpi
    return min(0.6, MAX_BAR_INCHES / (axes_inches / n_bars))


def _text_color_on(background: tuple[float, ...]) -> str:
    """背景色の明るさに応じて，読みやすい文字色を返す．

    Args:
        background: 背景色の RGBA（各成分は 0〜1）．カラーマップが返す形のタプル．

    Returns:
        暗い背景なら "white"，明るい背景なら INK_PRIMARY．
    """
    r, g, b = to_rgb(background)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.5 else INK_PRIMARY
