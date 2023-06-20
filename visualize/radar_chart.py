import os
import re
import sys
from typing import List, cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    data = sys.argv[1]
    if data == "empirical":
        targets = ["aps", "twitter"]
    elif data == "synthetic":
        targets = [
            f"{data}/rho5_nu5_sSSW",
            f"{data}/rho5_nu5_sWSW",
            f"{data}/rho5_nu15_sSSW",
            f"{data}/rho5_nu15_sWSW",
            f"{data}/rho20_nu7_sSSW",
            f"{data}/rho20_nu7_sWSW",
        ]
        synthetic = pd.read_csv("../data/synthetic_target.csv").set_index(["rho", "nu", "s"]).sort_index()
        synthetic_mean = synthetic.groupby(["rho", "nu", "s"]).mean()
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

    fm: matplotlib.font_manager.FontManager = matplotlib.font_manager.fontManager
    fm.addfont("./STIXTwoText.ttf")
    plt.rcParams["font.family"] = "STIX Two Text"
    plt.rcParams["font.size"] = 16

    os.makedirs(f"results/radar_chart", exist_ok=True)

    for target in targets:
        if data == "empirical":
            target_data = pd.read_csv(f"../data/{target}.csv").iloc[0].sort_index()
        else:
            os.makedirs(f"results/radar_chart/synthetic", exist_ok=True)
            pattern = r"synthetic/rho(\d+)_nu(\d+)_s(\w+)"
            matches = re.match(pattern, target)
            if matches:
                rho = int(matches.group(1))
                nu = int(matches.group(2))
                s = matches.group(3)
            target_data = synthetic_mean.loc[(rho, nu, s), :].sort_index()

        fs_results = (
            pd.read_csv(f"../full-search/results/existing_full_search.csv").set_index(["rho", "nu", "s"]).sort_index()
        )
        fs_results_mean = fs_results.groupby(by=["rho", "nu", "s"]).mean()

        fs_best_params = (fs_results_mean - target_data).abs().sum(axis=1).idxmin()
        fs_best_vec = fs_results.loc[fs_best_params, :].mean().sort_index()
        qd_best_vec = pd.read_csv(f"results/fitted/{target}/qd.csv").mean().sort_index()
        ga_best_vec = pd.read_csv(f"results/fitted/{target}/ga.csv").mean().sort_index()

        labels = cast(List[str], list(target_data.index) + [target_data.index[0]])
        labels = list(map(lambda l: readable_metrics[l], labels))

        target_values = list(target_data.values) + [target_data.values[0]]
        fs_values = list(fs_best_vec.values) + [fs_best_vec.values[0]]
        qd_values = list(qd_best_vec.values) + [qd_best_vec.values[0]]
        ga_values = list(ga_best_vec.values) + [ga_best_vec.values[0]]

        theta = np.linspace(0, np.pi * 2, len(labels))

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(theta, fs_values, label="existing", color="#FC8484")
        ax.plot(theta, ga_values, label="Genetic Algorithm", color="#76ABCB")
        ax.plot(theta, qd_values, label="proposed", color="#51BD56")
        ax.plot(theta, target_values, label="target", color="#505050", linestyle="dashed")
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=20)
        ax.set_ylim(0, 1)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(f"results/radar_chart/{target}.png", dpi=300)
        plt.show()
        plt.close()
