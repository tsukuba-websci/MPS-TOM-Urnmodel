import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_box(target_type: str, targets: list, my_color: dict):
    algorithms = ["full-search", "qd", "ga", "random-search"]
    palette = {
        "full-search": my_color["red"],
        "qd": my_color["light_green"],
        "ga": my_color["light_blue"],
        "random-search": my_color["purple"],
    }
    if target_type == "empirical":
        target_labels = {
            "twitter": "TMN",
            "aps": "APS",
            "mixi": "MIXI",
        }
    else:
        target_labels = {
            f"{target_type}/rho5_nu5_sSSW": "(5,5,SSW)",
            f"{target_type}/rho5_nu5_sWSW": "(5,5,WSW)",
            f"{target_type}/rho5_nu15_sSSW": "(5,15,SSW)",
            f"{target_type}/rho5_nu15_sWSW": "(5,15,WSW)",
            f"{target_type}/rho20_nu7_sSSW": "(20,7,SSW)",
            f"{target_type}/rho20_nu7_sWSW": "(20,7,WSW)",
        }
    if target_type == "empirical":
        generation = 500
    else:
        generation = 100

    data_list = []

    for target in targets:
        for algorithm in algorithms:
            if algorithm == "ga" or algorithm == "qd":
                filenum_str = "{:0>8}".format(generation - 1)
                file_path = f"../{algorithm}/results/{target}/archives/{filenum_str}.csv"
            elif algorithm == "full-search" or algorithm == "random-search":
                file_path = f"../{algorithm}/results/{target}/archive.csv"
            try:
                data = pd.read_csv(file_path)
                data["algorithm"] = algorithm
                data["target"] = target_labels[target]
                data_list.append(data)
            except FileNotFoundError:
                print(f"File not found: {file_path}")

    combined_data = pd.concat(data_list)

    plt.figure(figsize=(10, 5))
    sns.boxplot(
        x="target",
        y="distance",
        hue="algorithm",
        data=combined_data,
        palette=palette,
        fliersize=2,
    )
    os.makedirs("results/box", exist_ok=True)
    plt.legend().remove()
    plt.ylabel("d")
    plt.xlabel("")
    plt.savefig(f"results/box/{target_type}.png")
    plt.show()
