import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

color = ["#ff7f0e", "#1f77b4", "#9CDAA0"]


def archives2df(df, df_min, target, algorithm):
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


def plot(df, df_min):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df_min,
        x="generation",
        y="distance",
        hue="target",
        legend=False,
        linestyle="--",
        ax=ax,
        palette=color,
    )
    sns.lineplot(
        data=df,
        x="generation",
        y="distance",
        hue="target",
        legend=False,
        ax=ax,
        alpha=0.3,
        palette=color,
    )


def plot_timeline(data: str, targets) -> None:
    algorithms = ["ga", "qd"]

    for algorithm in algorithms:
        # 実データに対しては、全てのターゲットをまとめてプロット
        if data == "empirical":
            os.makedirs(f"results/timeline/{algorithm}", exist_ok=True)

            df = pd.DataFrame()
            df_min = pd.DataFrame()
            for target in targets:
                df, df_min = archives2df(df, df_min, target, algorithm)
            plot(df, df_min)

            plt.xlabel("Generation")
            plt.ylabel("d")
            plt.tight_layout()
            plt.savefig(f"results/timeline/{algorithm}/{data}.png", dpi=300)
            plt.close()

        # 合成データに対しては、ターゲットごとにプロット
        else:
            os.makedirs(f"results/timeline/{algorithm}/synthetic", exist_ok=True)

            for target in targets:
                df = pd.DataFrame()
                df_min = pd.DataFrame()
                df, df_min = archives2df(df, df_min, target, algorithm)
                plot(df, df_min)

                plt.xlabel("Generation")
                plt.ylabel("d")
                plt.tight_layout()
                plt.savefig(f"results/timeline/{algorithm}/{target}.png", dpi=300)
                plt.close()
