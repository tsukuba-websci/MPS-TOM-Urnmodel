import argparse
import pandas as pd


def export_latex_table(targets: list) -> None:
    algorithms = ["full-search", "qd", "ga", "random-search"]

    target_data = {}
    distances = {}

    for target in targets:
        target_data[target] = pd.read_csv(f"../data/{target}.csv").iloc[0]

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

    print(dfs.reset_index(drop=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_type", type=str, choices=["empirical", "synthetic"], help="データの種類")
    args = parser.parse_args()
    target_type = args.target_type

    if target_type == "empirical":
        # targets = ["twitter", "aps"]
        targets = ["mixi", "aps", "twitter"]
    elif target_type == "synthetic":
        # FIXME: 最良のときのターゲットを指定して下さい
        targets = ["synthetic/rho5_nu15_sSSW"]

    export_latex_table(targets)
