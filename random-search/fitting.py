import argparse
import os
import re

import pandas as pd


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("target_type", type=str, choices=["empirical", "synthetic"], help="データの種類")
    args = parser.parse_args()
    return args


def set_targets(target_type: str) -> list[str]:
    if target_type == "empirical":
        targets = ["aps", "twitter"]
    else:
        targets = [
            f"{target_type}/rho5_nu5_sSSW",
            f"{target_type}/rho5_nu5_sWSW",
            f"{target_type}/rho5_nu15_sSSW",
            f"{target_type}/rho5_nu15_sWSW",
            f"{target_type}/rho20_nu7_sSSW",
            f"{target_type}/rho20_nu7_sWSW",
        ]
    return targets


def set_target_data(target_type: str, target: str) -> pd.DataFrame:
    if target_type == "empirical":
        target_data = pd.read_csv(f"../data/{target}.csv").iloc[0]
    else:
        pattern = r"synthetic/rho(\d+)_nu(\d+)_s(\w+)"
        matches = re.match(pattern, target)
        if matches:
            rho = int(matches.group(1))
            nu = int(matches.group(2))
            s = matches.group(3)
        synthetic = pd.read_csv("../data/synthetic_target.csv").set_index(["rho", "nu", "s"]).sort_index()
        target_data = synthetic.loc[(rho, nu, s), :].mean()
    return target_data


def read_results(target_type: str, index: list) -> pd.DataFrame:
    rs_results = pd.read_csv("results/random_search.csv").set_index(index)

    if target_type == "synthetic":
        rs_results = rs_results.head(100)
    return rs_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    target_type = args.target_type

    index = ["rho", "nu", "recentness", "frequency"]

    targets = set_targets(target_type)

    for target in targets:
        target_data = set_target_data(target_type, target)

        rs_results = read_results(target_type, index)

        # find best params
        rs_results["distance"] = (rs_results - target_data).abs().sum(axis=1)
        rs_best_params = rs_results["distance"].idxmin()
        rs_best_distance = rs_results["distance"].min()

        # convert to pandas.DataFrame
        df = rs_results.reset_index()
        df_best = pd.DataFrame([rs_best_params], columns=index)
        df_best["distance"] = rs_best_distance

        # save params and distance
        dir = f"results/{target}"
        os.makedirs(dir, exist_ok=True)

        df[["rho", "nu", "recentness", "frequency", "distance"]].to_csv(f"{dir}/archive.csv", index=False)
        df_best.to_csv(f"{dir}/best.csv", index=False)
