import os
import sys

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
    data = sys.argv[1]
    if data == "empirical":
        targets = ["aps", "twitter"]
    # elif data == "synthetic":
    #     targets = [
    #         f"{data}/rho5_nu5_sSSW",
    #         f"{data}/rho5_nu5_sWSW",
    #         f"{data}/rho5_nu15_sSSW",
    #         f"{data}/rho5_nu15_sWSW",
    #         f"{data}/rho20_nu7_sSSW",
    #         f"{data}/rho20_nu7_sWSW",
    #     ]
    else:
        raise ValueError("must be 'synthetic' or 'empirical'")

    readable_metrics = {
        "gamma": "γ",
        "no": "NO",
        "nc": "NC",
        "oo": "OO",
        "oc": "OC",
        "c": "C",
        "y": "Y",
        "g": "G",
        "r": "R",
        "h": "<h>",
    }
    color = ["#ff7f0e", "#1f77b4"]

    fm: matplotlib.font_manager.FontManager = matplotlib.font_manager.fontManager
    fm.addfont("./STIXTwoText.ttf")
    plt.rcParams["font.family"] = "STIX Two Text"

    algorithms = ["ga", "qd"]

    for algorithm in algorithms:
        df = pd.DataFrame()
        df_min = pd.DataFrame()
        for target in targets:
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

        plt.xlabel("Generation")
        plt.ylabel("d")
        plt.tight_layout()
        plt.savefig(f"results/{algorithm}_timeline.png", dpi=300)
        plt.close()
