import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ribs.archives import CVTArchive
from sklearn.manifold import TSNE


def plot_map(targets: list, my_color: dict) -> None:
    my_green_to_red_cmap = LinearSegmentedColormap.from_list(
        "my_green_to_red_gradient", [my_color["dark_red"], my_color["yellow"], my_color["dark_green"]]
    )

    for target in targets:
        with open(f"../qd/results/{target}/archive.pkl", "rb") as f:
            archive: CVTArchive = pickle.load(f)
        df = archive.as_pandas()

        # graphのベクトルの抽出
        dim = df.filter(regex="measure_\d+").shape[1]
        column_names = [f"measure_{i}" for i in range(dim)]
        data = df[column_names]

        # 正規化
        normalized_data = (data - data.mean()) / data.std()

        # t-SNEで次元削減
        tsne = TSNE(n_components=2)  # 削減後の次元数を2に設定
        reduced_data = tsne.fit_transform(normalized_data)

        # reduced_dataをDataFrameに変換
        reduced_data_df = pd.DataFrame(reduced_data, columns=["Dimension 1", "Dimension 2"])

        # "distance"列を追加
        reduced_data_df["distance"] = df["objective"].abs()

        # 可視化
        os.makedirs("results/qd_map", exist_ok=True)
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(
            reduced_data_df["Dimension 1"],
            reduced_data_df["Dimension 2"],
            c=reduced_data_df["distance"],
            cmap=my_green_to_red_cmap,
        )
        cbar = plt.colorbar(scatter, label="Distance", orientation="vertical")
        cbar.ax.invert_yaxis()
        plt.xlabel("dimension 1")
        plt.ylabel("dimension 2")
        plt.savefig(f"results/qd_map/{target}.png")


if __name__ == "__main__":
    targets = ["twitter", "aps"]
    # targets = ["twitter", "aps", "mixi"]

    my_color = {
        "red": "#FC8484",
        "dark_red": "#FA5050",
        "light_blue": "#94C4E0",
        "light_green": "#9CDAA0",
        "dark_blue": "#76ABCB",
        "dark_green": "#51BD56",
        "black": "#505050",
        "purple": "#CBA6DD",
        "yellow": "#FFE959",
    }

    plot_map(targets, my_color)
