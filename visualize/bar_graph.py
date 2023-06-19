import sys

import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib import pyplot as plt

my_red = "#FC8484"
my_green = "#9CDAA0"
my_blue = "#9CC3DA"

if __name__ == "__main__":
    data = sys.argv[1]
    if data == "empirical":
        targets = ["twitter", "aps"]
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
    plt.rcParams["axes.prop_cycle"] = cycler(color=["#FC8484", "#9CDAA0", "#F5C08B", "#F7E393"])

    df = pd.DataFrame()
    fs_results = (
        pd.read_csv("../full-search/results/existing_full_search.csv").set_index(["rho", "nu", "s"]).sort_index()
    )
    fs_results_mean = fs_results.groupby(by=["rho", "nu", "s"]).mean()
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
        hue="model",
        palette={"Existing": my_red, "Proposed": my_green, "GA": my_blue},
        legend=False,  # type: ignore
    )
    plt.ylabel("d")
    plt.xlabel("")
    plt.savefig("results/distances.png", dpi=300)
    plt.show()
    plt.close()
