import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ribs.archives import CVTArchive
from sklearn.manifold import TSNE


def plot_qd_map(target_type: str, targets: list, my_color: dict) -> None:
    if target_type != "empirical":
        print("plot_qd_map only supports empirical data.")
        return

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
        reduced_data_df = pd.DataFrame(reduced_data, columns=["t-sne1", "t-sne2"])

        # "distance"列を追加
        reduced_data_df["distance"] = df["objective"].abs()

        # 可視化
        os.makedirs("results/qd_map", exist_ok=True)
        plt.rcParams["font.size"] = 22
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            reduced_data_df["t-sne1"],
            reduced_data_df["t-sne2"],
            c=reduced_data_df["distance"],
            cmap=my_green_to_red_cmap,
        )
        cbar = plt.colorbar(scatter, label="d", orientation="vertical", shrink=1)
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=16)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tick_params(axis="both", labelsize=18)
        plt.tight_layout()
        plt.savefig(f"results/qd_map/{target}.png", dpi=300)
        plt.close()
