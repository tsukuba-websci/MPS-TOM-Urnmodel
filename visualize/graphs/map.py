import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE


def plot(df: pd.DataFrame, cmap: LinearSegmentedColormap, file: str, xlim: tuple, ylim: tuple, vlim: tuple) -> None:
    dir = os.path.dirname(file)
    os.makedirs(dir, exist_ok=True)
    plt.rcParams["font.size"] = 30
    plt.figure(figsize=(10, 8))

    # distanceが小さいほどマーカーを大きくする
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
        vmax=vlim[1],
        vmin=vlim[0],
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
    normalized_data = (data - data.mean(axis=0)) / data.std(axis=0)

    # t-SNEで次元削減
    tsne = TSNE(n_components=2)  # 削減後の次元数を2に設定
    reduced_data = tsne.fit_transform(normalized_data)

    # reduced_dataをDataFrameに変換
    reduced_data_df = pd.DataFrame(reduced_data, columns=["t-sne1", "t-sne2"])

    return reduced_data_df


def read_data(target_type: str, targets: list, type: str):
    if target_type == "empirical":
        generation = 500
    else:
        generation = 100
    df = pd.DataFrame()

    for target in targets:
        for algorithm in ["qd", "ga"]:
            file_path = f"results/vec/{target}/{algorithm}.csv"
            df_ = pd.read_csv(file_path)
            df_["target"] = target
            df_["algorithm"] = algorithm
            df = pd.concat([df, df_])
    return df


def phenotype_map(target_type: str, targets: list, my_color: dict) -> None:
    my_green_to_red_cmap = LinearSegmentedColormap.from_list(
        "my_green_to_red_gradient",
        [my_color["dark_red"], my_color["yellow"], my_color["yellow_green"], my_color["dark_green"]],
    )

    df = read_data(target_type, targets, "phenotype")

    column_names = [f"vec{i}" for i in range(128)]
    vec = df[column_names].values

    # 次元削減
    reduced_vec = reduce_dimension(vec)

    # dfにreduced_vecを追加
    df.reset_index(drop=True, inplace=True)
    df[["t-sne1", "t-sne2"]] = reduced_vec

    # x軸とy軸の範囲を設定
    margin = 5
    xlim = (df["t-sne1"].min() - margin, df["t-sne1"].max() + margin)
    ylim = (df["t-sne2"].min() - margin, df["t-sne2"].max() + margin)

    for target in targets:
        # colorbarの範囲を設定
        vmin = df[df["target"] == target]["distance"].min()
        vmax = df[df["target"] == target]["distance"].max()
        # 可視化
        for algorithm in ["qd", "ga"]:
            file = f"results/map/phenotype/{target}_{algorithm}.png"
            df_target_algo = df[(df["target"] == target) & (df["algorithm"] == algorithm)]
            plot(df_target_algo, my_green_to_red_cmap, file, xlim, ylim, (vmin, vmax))
