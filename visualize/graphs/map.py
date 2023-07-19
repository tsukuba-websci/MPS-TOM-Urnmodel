import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ribs.archives import CVTArchive
from sklearn.manifold import TSNE


def plot(df: pd.DataFrame, cmap: LinearSegmentedColormap, file: str, xlim, ylim, vmin, vmax) -> None:
    dir = os.path.dirname(file)
    os.makedirs(dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    min_value = df["distance"].min()
    max_value = df["distance"].max()
    s = (df["distance"] - min_value) * (100 - 15) / (max_value - min_value) + 15
    s = 115 - s
    scatter = plt.scatter(
        df["t-sne1"],
        df["t-sne2"],
        c=df["distance"],
        cmap=cmap,
        s=s,
        alpha=0.5,
        vmax=vmax,
        vmin=vmin,
    )
    cbar = plt.colorbar(scatter, label="d", orientation="vertical", shrink=1)
    cbar.ax.invert_yaxis()
    cbar.ax.tick_params(labelsize=22)
    plt.xlim(xlim)
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


def read_data(target_type: str, targets: list):
    if target_type == "empirical":
        generation = 500
    else:
        generation = 100
    df = pd.DataFrame()
    for target in targets:
        for algorithm in ["qd", "ga"]:
            filenum_str = "{:0>8}".format(generation - 1)
            file_path = f"../{algorithm}/results/{target}/archives/{filenum_str}.csv"
            df_ = pd.read_csv(file_path)
            df_["target"] = target
            df_["algorithm"] = algorithm
            df = pd.concat([df, df_])
    return df


def genotype_map(target_type: str, targets: list, my_color: dict) -> None:
    my_green_to_red_cmap = LinearSegmentedColormap.from_list(
        "my_green_to_red_gradient",
        [my_color["dark_red"], my_color["yellow"], my_color["yellow_green"], my_color["dark_green"]],
    )

    df = read_data(target_type, targets)
    print(df)

    # 次元削減
    vec = df[["rho", "nu", "recentness", "frequency"]]
    reduced_vec = reduce_dimension(vec)

    # dfにreduced_vecを追加
    df[["t-sne1", "t-sne2"]] = reduced_vec

    margin = 5
    xlim = (df["t-sne1"].min() - margin, df["t-sne1"].max() + margin)
    ylim = (df["t-sne2"].min() - margin, df["t-sne2"].max() + margin)

    for target in targets:
        vmin = df[df["target"] == target]["distance"].min()
        vmax = df[df["target"] == target]["distance"].max()
        # 可視化
        for algorithm in ["qd", "ga"]:
            file = f"results/map/genotype/{target}_{algorithm}.png"
            df_target_algo = df[(df["target"] == target) & (df["algorithm"] == algorithm)]
            plot(df_target_algo, my_green_to_red_cmap, file, xlim, ylim, vmin, vmax)
