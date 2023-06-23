import os
import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_bar_graph(target_type: str, targets, my_color) -> None:
    df = pd.DataFrame()
    fs_results = (
        pd.read_csv("../full-search/results/existing_full_search.csv").set_index(["rho", "nu", "s"]).sort_index()
    )
    fs_results_mean = fs_results.groupby(by=["rho", "nu", "s"]).mean()

    os.makedirs("results/bar_graph", exist_ok=True)

    if target_type == "empirical":
        for target in targets:
            emp = pd.read_csv(f"../data/{target}.csv").iloc[0]

            fs_best_params = (fs_results_mean - emp).abs().sum(axis=1).idxmin()
            fs_best_vecs = fs_results.loc[fs_best_params, :]
            fs_best_diffs = (fs_best_vecs - emp).abs().sum(axis=1)
            fs_df = pd.DataFrame(data=fs_best_diffs, columns=["distance"])
            fs_df["target"] = target
            fs_df["model"] = "Existing"

            qd_best_vecs = pd.read_csv(f"results/fitted/{target}/qd.csv")
            qd_best_diffs = (qd_best_vecs - emp).abs().sum(axis=1)
            qd_df = pd.DataFrame(data=qd_best_diffs, columns=["distance"])
            qd_df["target"] = target
            qd_df["model"] = "Proposed"

            ga_best_vecs = pd.read_csv(f"results/fitted/{target}/ga.csv")
            ga_best_diffs = (ga_best_vecs - emp).abs().sum(axis=1)
            ga_df = pd.DataFrame(data=ga_best_diffs, columns=["distance"])
            ga_df["target"] = target
            ga_df["model"] = "GA"

            df = pd.concat([df, fs_df, qd_df, ga_df], ignore_index=True)

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
            palette={"Existing": my_color["red"], "Proposed": my_color["light_green"], "GA": my_color["light_blue"]},
            legend=False,  # type: ignore
        )
        plt.ylabel("d")
        plt.xlabel("")
        plt.savefig("results/bar_graph/empirical.png", dpi=300)
        plt.close()

    else:
        df = pd.DataFrame()

        targets_data = pd.read_csv("../data/synthetic_target.csv").set_index(["rho", "nu", "s"]).sort_index()
        targets_mean = targets_data.groupby(["rho", "nu", "s"]).mean()

        pattern = r"synthetic/rho(\d+)_nu(\d+)_s(SSW|WSW)"

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

            row = pd.Series(
                {
                    "rho": rho,
                    "nu": nu,
                    "s": s,
                    "fs_mean": fs_mean,
                    "qd_mean": qd_mean,
                    "ga_mean": ga_mean,
                }
            )
            df = pd.concat([df, row.to_frame().T], ignore_index=True)

        plt.rcParams["font.size"] = 16
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            x=["Full Search", "Quality Diversity", "GA"],
            height=df[["fs_mean", "qd_mean", "ga_mean"]].mean(),
            yerr=df[["fs_mean", "qd_mean", "ga_mean"]].std(),
            color=[my_color["red"], my_color["light_green"], my_color["light_blue"]],
            capsize=4,
        )
        plt.ylabel("d")
        plt.tight_layout()
        plt.savefig("results/bar_graph/synthetic.png", dpi=300)
        plt.close()
