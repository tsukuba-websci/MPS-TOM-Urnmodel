import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams["font.size"] = 18


def archives2df(df: pd.DataFrame, df_min: pd.DataFrame, target: str, algorithm: str):
    basedir = f"../{algorithm}/results/{target}/archives"
    files = sorted(os.listdir(basedir))

    for gen, file in enumerate(files):
        _df = pd.read_csv(f"{basedir}/{file}")
        _df["generation"] = gen
        _df["target"] = target
        _df = _df[["target", "generation", "distance"]]
        df = pd.concat([df, _df])

        _df_min = _df.head(1)
        df_min = pd.concat([df_min, _df_min])
    return df, df_min


def plot_data(algorithms: list, data: dict, data_min: dict, ymax: float, file_name: str, palette: list) -> None:
    for algorithm in algorithms:
        fig, ax = plt.subplots(figsize=(8, 5))
        # 全個体の距離の平均と標準偏差
        sns.lineplot(
            data=data[algorithm],
            x="generation",
            y="distance",
            hue="target",
            legend=False,
            ax=ax,
            alpha=0.3,
            palette=palette,
        )
        # 各世代の最良個体の距離
        sns.lineplot(
            data=data_min[algorithm],
            x="generation",
            y="distance",
            hue="target",
            legend=False,
            ax=ax,
            linestyle="--",
            palette=palette,
        )
        plt.xlabel("Generation", fontsize=24)
        plt.ylabel("d", fontsize=24)
        plt.ylim(0, ymax)
        plt.tight_layout()
        plt.savefig(f"results/timeline/{algorithm}/{file_name}.png", dpi=300)
        plt.close()


def plot_timeline(target_type: str, targets: list, my_color: dict) -> None:
    algorithms = ["ga", "qd"]
    if target_type == "empirical":
        data = {}
        data_min = {}
        ymax = 0
        for algorithm in algorithms:
            data_alg = pd.DataFrame()
            data_alg_min = pd.DataFrame()

            for target in targets:
                df, df_min = archives2df(pd.DataFrame(), pd.DataFrame(), target, algorithm)
                data_alg = pd.concat([data_alg, df])
                data_alg_min = pd.concat([data_alg_min, df_min])
                ymax = max(ymax, df.groupby("generation")["distance"].mean().max())
            data[algorithm] = data_alg
            data_min[algorithm] = data_alg_min

        plot_data(algorithms, data, data_min, ymax, target_type, None)

    else:
        for target in targets:
            data = {}
            data_min = {}
            ymax = 0
            for algorithm in algorithms:
                data_alg = pd.DataFrame()
                data_alg_min = pd.DataFrame()

                df, df_min = archives2df(pd.DataFrame(), pd.DataFrame(), target, algorithm)
                data_alg = pd.concat([data_alg, df])
                data_alg_min = pd.concat([data_alg_min, df_min])
                ymax = max(ymax, df.groupby("generation")["distance"].mean().max())
                data[algorithm] = data_alg
                data_min[algorithm] = data_alg_min

            plot_data(algorithms, data, data_min, ymax, target, [my_color["dark_red"]])
