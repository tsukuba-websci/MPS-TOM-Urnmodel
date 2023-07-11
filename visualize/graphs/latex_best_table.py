import argparse
import os

import pandas as pd

# NOTE: MPS-TOM-urnmodel/で用意されているpoetry環境ではpandasのversionの依存関係が合わず、動きません。
# pandas>=1.5.3, Jinja2>=3.0.0, matplotlib, numpyのある環境で実行してください。


def export_latex_table(target_type: str, targets: list) -> None:
    contents = "\\begin{table*}[h]\n"

    dfs = []

    algorithms = ["full-search", "qd", "ga", "random-search"]
    targets = ["twitter", "aps"]

    for algorithm in algorithms:
        for target in targets:
            print(algorithm, target)
            if algorithm == "full-search":
                file_path = f"../{algorithm}/results/{target}/archive.csv"
                data = pd.read_csv(file_path)
                distance = data.groupby(by=["rho", "nu", "recentness", "frequency"]).mean().min().values[0]
            else:
                file_path = f"../{algorithm}/results/{target}/best.csv"
                data = pd.read_csv(file_path)
                distance = data["distance"].values[0]
            df = pd.DataFrame({"algorithm": [algorithm], "target": [target], "distance": [distance]})
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    table = df.pivot(index="target", columns="algorithm", values="distance")
    print(table)

    l = (
        df.style.hide()
        .relabel_index(
            labels=[r"$\#$", r"$Existing Method$", r"$Quality Diversity$", r"$Genetic Alogorithm$" r"$Random Search$"],
            axis=1,
        )
        .set_properties(**{"color": "black"})
        .to_latex(
            convert_css=True,
            clines="all;data",
            hrules=True,
        )
    )
    table = [
        rf"\caption{{best distance}}",
        rf"\label{{table:best_distance}}",
        l.removesuffix("\n"),
        r"\hfill",
    ]
    contents += "\n".join(table)

    contents += "\\end{table*}"

    os.makedirs("./results/table", exist_ok=True)
    with open(f"./results/table/best_distance.tex", "w") as f:
        f.write(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_type", type=str, choices=["empirical", "synthetic"], help="データの種類")
    args = parser.parse_args()
    target_type = args.target_type

    if target_type == "empirical":
        targets = ["twitter", "aps"]
        # targets = ["twitter", "aps", "mixi"]
    elif target_type == "synthetic":
        # FIXME: 最良のときのターゲットを指定して下さい
        targets = ["synthetic/rho5_nu15_sSSW"]
        generation = 100

    export_latex_table(target_type, targets)
