import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import _mathtext as mathtext
from matplotlib.colors import LinearSegmentedColormap
from ribs.archives import CVTArchive
from sklearn.manifold import TSNE


def plot(reduced_data_df: pd.DataFrame, cmap: LinearSegmentedColormap, file: str, xlim=None, ylim=None) -> None:
    dir = os.path.dirname(file)
    os.makedirs(dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    min_value = reduced_data_df["distance"].min()
    max_value = reduced_data_df["distance"].max()
    s = (reduced_data_df["distance"] - min_value) * (100 - 15) / (max_value - min_value) + 15
    s = 115 - s
    scatter = plt.scatter(
        reduced_data_df["t-sne1"],
        reduced_data_df["t-sne2"],
        c=reduced_data_df["distance"],
        cmap=cmap,
        s=s,
        alpha=0.5,
    )
    cbar = plt.colorbar(scatter, label="d", orientation="vertical", shrink=1)
    cbar.ax.invert_yaxis()
    cbar.ax.tick_params(labelsize=22)
    plt.xlabel("$t-SNE_1$")
    plt.ylabel("$t-SNE_2$")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tick_params(axis="both", labelsize=22)
    plt.tight_layout()
    plt.savefig(file, dpi=200)
    plt.close()


def reduce_dimension(data: pd.DataFrame) -> pd.DataFrame:
    # 正規化
    normalized_data = (data - data.mean()) / data.std()

    # t-SNEで次元削減
    tsne = TSNE(n_components=2)  # 削減後の次元数を2に設定
    reduced_data = tsne.fit_transform(normalized_data)

    # reduced_dataをDataFrameに変換
    reduced_data_df = pd.DataFrame(reduced_data, columns=["t-sne1", "t-sne2"])

    return reduced_data_df


def phenotype_map(df: pd.DataFrame, target: str, cmap: LinearSegmentedColormap) -> None:
    # graphのベクトルの抽出
    dim = df.filter(regex="measure_\d+").shape[1]
    column_names = [f"measure_{i}" for i in range(dim)]
    data = df[column_names]

    # 次元削減
    reduced_data_df = reduce_dimension(data)

    # "distance"列を追加
    reduced_data_df["distance"] = df["objective"].abs()

    # 可視化
    file = f"results/qd_map/phenotype/{target}.png"
    plot(reduced_data_df, cmap, file)


def genotype_map(
    df: pd.DataFrame, targets: list, cmap: LinearSegmentedColormap, population_size, distances, data
) -> None:
    # 次元削減
    reduced_data_df = reduce_dimension(data)

    # "distance"列を追加
    reduced_data_df["distance"] = distances

    margin = 5
    xlim = (reduced_data_df["t-sne1"].min() - margin, reduced_data_df["t-sne1"].max() + margin)
    ylim = (reduced_data_df["t-sne2"].min() - margin, reduced_data_df["t-sne2"].max() + margin)

    end_index = 0
    for target, pop_size in zip(targets, population_size):
        # 可視化
        file = f"results/qd_map/genotype/{target}.png"
        plot(reduced_data_df[end_index : end_index + pop_size], cmap, file, xlim, ylim)
        end_index += pop_size


def plot_qd_map(targets: list, my_color: dict) -> None:
    # 下付き文字の設定
    mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants
    mathtext.FontConstantsBase.sub1 = 0.1

    plt.rcParams.update(
        {
            "mathtext.default": "default",
            "mathtext.fontset": "stix",
            "font.size": 30,
            "figure.figsize": (3, 3),
        }
    )
    my_green_to_red_cmap = LinearSegmentedColormap.from_list(
        "my_green_to_red_gradient",
        [my_color["dark_red"], my_color["yellow"], my_color["yellow_green"], my_color["dark_green"]],
    )
    data = pd.DataFrame()
    distances = pd.DataFrame()
    popuration_size = []

    for target in targets:
        with open(f"../qd/results/{target}/archive.pkl", "rb") as f:
            archive: CVTArchive = pickle.load(f)
        df = archive.as_pandas()

        # 遺伝子の抽出
        df.rename(
            columns={"solution_0": "rho", "solution_1": "nu", "solution_2": "recentness", "solution_3": "frequency"},
            inplace=True,
        )

        data_target = df[["rho", "nu", "recentness", "frequency"]]
        data = pd.concat([data, data_target])

        distances = pd.concat([distances, df["objective"].abs()])

        popuration_size.append(len(df))

        phenotype_map(df, target, my_green_to_red_cmap)

    distances.reset_index(drop=True, inplace=True)
    genotype_map(df, targets, my_green_to_red_cmap, popuration_size, distances, data)
