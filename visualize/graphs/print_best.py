import argparse
import re

import pandas as pd


def read_target_data(target_type: str, targets: list) -> dict:
def print_best(targets: list) -> None:
    algorithms = ["full-search", "qd", "ga", "random-search"]

    target_data = {}

    if target_type == "empirical":
        for target in targets:
            target_data[target] = pd.read_csv(f"../data/{target}.csv").iloc[0]
    else:
        for target in targets:
            pattern = r"synthetic/rho(\d+)_nu(\d+)_s(\w+)"
            matches = re.match(pattern, target)
            if matches:
                rho = int(matches.group(1))
                nu = int(matches.group(2))
                s = matches.group(3)
            synthetic = pd.read_csv("../data/synthetic_target.csv").set_index(["rho", "nu", "s"]).sort_index()
            target_data[target] = synthetic.loc[(rho, nu, s), :].mean()

    return target_data


def caluculate_distance(target_data: dict, algorithms: list, targets: list) -> dict[str, dict[str, pd.Series]]:
    distances = {}

    for target in targets:
        distances[target] = {}

        for algorithm in algorithms:
            if algorithm == "full-search":
                file_path = f"../{algorithm}/results/{target}/archive.csv"
                data = pd.read_csv(file_path)
                min_distance_row = data.iloc[data["distance"].idxmin()]
                best_params = min_distance_row[["rho", "nu", "recentness", "frequency"]]
                tmp = data[data[["rho", "nu", "recentness", "frequency"]].eq(best_params).all(axis=1)]
                distance = tmp["distance"]

            else:
                file_path = f"results/fitted/{target}/{algorithm}.csv"
                data = pd.read_csv(file_path).head(10)
                distance = (data - target_data[target]).abs().sum(axis=1)

                file_best_csv = f"../{algorithm}/results/{target}/best.csv"
                data_best = pd.read_csv(file_best_csv)
                distance_best = data_best["distance"]
                distance = pd.concat([distance, distance_best], axis=0)

            distances[target][algorithm] = distance

    return distances


def create_dataframe(distances: dict, algorithms: list, targets: list) -> pd.DataFrame:
    dfs = pd.DataFrame()

    for target in targets:
        for algorithm in algorithms:
            df = pd.DataFrame(
                {
                    "algorithm": [algorithm],
                    "target": [target],
                    "mean": [distances[target][algorithm].mean()],
                    "std": [distances[target][algorithm].std()],
                }
            )
            dfs = pd.concat([dfs, df], axis=0)

    return dfs.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_type", type=str, choices=["empirical", "synthetic"], help="データの種類")
    args = parser.parse_args()
    target_type = args.target_type

    if target_type == "empirical":
        targets = ["twitter", "aps"]
        # targets = ["mixi", "aps", "twitter"]

    elif target_type == "synthetic":
        targets = [
            "synthetic/rho5_nu5_sSSW",
            "synthetic/rho5_nu5_sWSW",
            "synthetic/rho5_nu15_sSSW",
            "synthetic/rho5_nu15_sWSW",
            "synthetic/rho20_nu7_sSSW",
            "synthetic/rho20_nu7_sWSW",
        ]

    algorithms = ["full-search", "qd", "ga", "random-search"]

    target_data = read_target_data(target_type, targets)
    distances = caluculate_distance(target_data, algorithms, targets)
    dfs = create_dataframe(distances, algorithms, targets)

    print(dfs)
