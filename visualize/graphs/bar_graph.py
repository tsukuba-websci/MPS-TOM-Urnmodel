import os
import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

target_labels = {
    "twitter": "TMN",
    "aps": "APS",
    "mixi": "MIXI",
}


def plot_bar_graph(target_type: str, targets: list, my_color: dict) -> None:
    df = pd.DataFrame()
    fs_results = (
        pd.read_csv("../full-search/results/existing_full_search.csv").set_index(["rho", "nu", "s"]).sort_index()
    )
    fs_results_mean = fs_results.groupby(by=["rho", "nu", "s"]).mean()

    os.makedirs("results/bar_graph", exist_ok=True)

    if target_type == "empirical":
        df = pd.DataFrame(columns=["distance", "target", "model"])

        for target in targets:
            emp = pd.read_csv(f"../data/{target}.csv").iloc[0]

            algorithm_labels = [
                {"algorithm": "full-search", "model": "Existing"},
                {"algorithm": "qd", "model": "Proposed"},
                {"algorithm": "ga", "model": "GA"},
                {"algorithm": "random-search", "model": "Random Search"},
            ]

            for label in algorithm_labels:
                if label["algorithm"] == "full-search":
                    best_params = (fs_results_mean - emp).abs().sum(axis=1).idxmin()
                    best_vecs = fs_results.loc[best_params, :]
                else:
                    best_vecs = pd.read_csv(f"results/fitted/{target}/{label['algorithm']}.csv")
                best_diffs = (best_vecs - emp).abs().sum(axis=1)
                algorithm_df = pd.DataFrame(data=best_diffs, columns=["distance"])
                algorithm_df["target"] = target_labels[target]
                algorithm_df["model"] = label["model"]

                df = pd.concat([df, algorithm_df], ignore_index=True)

        g: sns.axisgrid.FacetGrid = sns.catplot(
            height=4,
            aspect=2,
            data=df,
            kind="bar",
            x="target",
            y="distance",
            errorbar="sd",
            capsize=0.02,
            errwidth=1.5,
            hue="model",
            palette={
                "Existing": my_color["red"],
                "Proposed": my_color["light_green"],
                "GA": my_color["light_blue"],
                "Random Search": my_color["purple"],
            },
            legend=False,  # type: ignore
        )
        plt.ylabel("d")
        plt.xlabel("")
        plt.savefig("results/bar_graph/empirical.png", dpi=300)
        plt.close()

    else:
        targets_data = pd.read_csv("../data/synthetic_target.csv").set_index(["rho", "nu", "s"]).sort_index()
        targets_mean = targets_data.groupby(["rho", "nu", "s"]).mean()

        pattern = r"synthetic/rho(\d+)_nu(\d+)_s(SSW|WSW)"

        results = []

        for target in targets:
            matches = re.match(pattern, target)
            if matches:
                rho = int(matches.group(1))
                nu = int(matches.group(2))
                s = matches.group(3)
                target_mean = targets_mean.loc[(rho, nu, s), :]
                fs_mean = (fs_results_mean - target_mean).dropna(axis=1).abs().sum(axis=1).min()

                qd_best_vecs = pd.read_csv(f"results/fitted/{target}/qd.csv")
                qd_mean = (qd_best_vecs - target_mean).abs().sum(axis=1).mean()

                ga_best_vecs = pd.read_csv(f"results/fitted/{target}/ga.csv")
                ga_mean = (ga_best_vecs - target_mean).abs().sum(axis=1).mean()

                rs_best_vecs = pd.read_csv(f"results/fitted/{target}/random-search.csv")
                rs_mean = (rs_best_vecs - target_mean).abs().sum(axis=1).mean()

                results.append(
                    {
                        "rho": rho,
                        "nu": nu,
                        "s": s,
                        "fs_mean": fs_mean,
                        "qd_mean": qd_mean,
                        "ga_mean": ga_mean,
                        "rs_mean": rs_mean,
                    }
                )

        df = pd.DataFrame(results)

        plt.rcParams["font.size"] = 13
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            x=["Full Search", "Quality Diversity", "GA", "Random Search"],
            height=df[["fs_mean", "qd_mean", "ga_mean", "rs_mean"]].mean(),
            yerr=df[["fs_mean", "qd_mean", "ga_mean", "rs_mean"]].std(),
            color=[my_color["red"], my_color["light_green"], my_color["light_blue"], my_color["purple"]],
            capsize=4,
        )
        plt.ylabel("d")
        plt.tight_layout()
        plt.savefig("results/bar_graph/synthetic.png", dpi=300)
        plt.close()
