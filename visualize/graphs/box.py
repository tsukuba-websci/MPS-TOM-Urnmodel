import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_target_labels(target_type: str) -> dict[str, str]:
    # 図に表示するターゲットデータのラベルを設定
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
    return target_labels


def load_data_for_algorithms(
    algorithms: list, algorithms_labels: dict, target: str, target_labels: dict, data_list: list, generation: int
) -> list:
    for algorithm in algorithms:
        # 結果のファイルパスを指定
        if algorithm == "ga" or algorithm == "qd":
            filenum_str = "{:0>8}".format(generation - 1)
            file_path = f"../{algorithm}/results/{target}/archives/{filenum_str}.csv"
        elif algorithm == "full-search" or algorithm == "random-search":
            file_path = f"../{algorithm}/results/{target}/archive.csv"

        # ファイルを読み込み、アルゴリズムとターゲットのラベルを付けて、リストに追加
        try:
            data = pd.read_csv(file_path)
            data["algorithm"] = algorithms_labels[algorithm]
            data["target"] = target_labels[target]
            data_list.append(data)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
    return data_list


def plot_box(target_type: str, targets: list, my_color: dict):
    algorithms = ["full-search", "qd", "ga", "random-search"]
    algorithms_labels = {
        "full-search": "Existing Method",
        "qd": "Quality Diversity",
        "ga": "Genetic Algorithm",
        "random-search": "Random Search",
    }
    palette = {
        "Existing Method": my_color["red"],
        "Quality Diversity": my_color["light_green"],
        "Genetic Algorithm": my_color["light_blue"],
        "Random Search": my_color["purple"],
    }
    target_labels = set_target_labels(target_type)

    if target_type == "empirical":
        # 実データの場合は、全ターゲットをまとめてプロットする
        generation = 500

        data_list = []
        for target in targets:
            data_list = load_data_for_algorithms(
                algorithms, algorithms_labels, target, target_labels, data_list, generation
            )

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
        plt.tight_layout()
        plt.savefig(f"results/box/{target_type}.png", dpi=300)

    else:
        # 合成データの場合は、ターゲットごとにプロットする
        generation = 100

        for target in targets:
            data_list = []
            data_list = load_data_for_algorithms(
                algorithms, algorithms_labels, target, target_labels, data_list, generation
            )
            combined_data = pd.concat(data_list)

            plt.figure(figsize=(8, 5))
            plt.rcParams["font.size"] = 15
            sns.boxplot(
                x="algorithm",
                y="distance",
                data=combined_data,
                palette=palette,
                fliersize=2,
                width=0.5,
            )
            os.makedirs(f"results/box/{target_type}", exist_ok=True)
            plt.tick_params(axis="both")
            plt.ylabel("d")
            plt.xlabel("")
            plt.tight_layout()
            plt.savefig(f"results/box/{target}.png", dpi=300)
