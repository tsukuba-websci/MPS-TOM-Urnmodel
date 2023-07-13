import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ribs.archives import CVTArchive
from sklearn.manifold import TSNE


def plot(reduced_data_df: pd.DataFrame, cmap: LinearSegmentedColormap, file: str) -> None:
    dir = os.path.dirname(file)
    os.makedirs(dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    s = (reduced_data_df["distance"].max() - reduced_data_df["distance"] + 0.01) * 100
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
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
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


def genotype_map(df: pd.DataFrame, target: str, cmap: LinearSegmentedColormap) -> None:
    # 遺伝子の抽出
    df.rename(
        columns={"solution_0": "rho", "solution_1": "nu", "solution_2": "recentness", "solution_3": "frequency"},
        inplace=True,
    )
    data = df[["rho", "nu", "recentness", "frequency"]]

    # 次元削減
    reduced_data_df = reduce_dimension(data)

    # "distance"列を追加
    reduced_data_df["distance"] = df["objective"].abs()

    # 可視化
    file = f"results/qd_map/genotype/{target}.png"
    plot(reduced_data_df, cmap, file)


def plot_qd_map(target_type: str, targets: list, my_color: dict) -> None:
    my_green_to_red_cmap = LinearSegmentedColormap.from_list(
        "my_green_to_red_gradient", [my_color["dark_red"], my_color["yellow"], my_color["dark_green"]]
    )

    for target in targets:
        with open(f"../qd/results/{target}/archive.pkl", "rb") as f:
            archive: CVTArchive = pickle.load(f)
        df = archive.as_pandas()

        phenotype_map(df, target, my_green_to_red_cmap)
        genotype_map(df, target, my_green_to_red_cmap)
