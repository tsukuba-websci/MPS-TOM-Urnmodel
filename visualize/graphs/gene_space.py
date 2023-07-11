import argparse
import os

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("graph_type", type=str, choices=["bar", "radar", "timeline", "box"], help="グラフの種類")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_type", type=str, choices=["empirical", "synthetic"], help="データの種類")
    args = parser.parse_args()
    target_type = args.target_type

    if target_type == "empirical":
        targets = ["twitter", "aps"]
        # targets = ["twitter", "aps", "mixi"]
        generation = 500
    elif target_type == "synthetic":
        # FIXME: 最良のときのターゲットを指定して下さい
        targets = ["synthetic/rho5_nu15_sSSW"]
        generation = 100

    fm: matplotlib.font_manager.FontManager = matplotlib.font_manager.fontManager
    fm.addfont("./STIXTwoText.ttf")
    plt.rcParams["font.family"] = "STIX Two Text"

    my_color = {
        "dark_red": "#FA5050",
        "dark_green": "#51BD56",
        "dark_blue": "#76ABCB",
        "yellow": "#E4E4E4",
    }
    my_red_cmap = LinearSegmentedColormap.from_list("my_red_gradient", colors=["red", "white"])
    my_green_cmap = LinearSegmentedColormap.from_list("my_green_gradient", colors=["green", "white"])
    my_blue_cmap = LinearSegmentedColormap.from_list("my_blue_gradient", colors=["blue", "white"])
    my_purple_cmap = LinearSegmentedColormap.from_list("my_purple_gradient", colors=["purple", "white"])

    cmap_dict = {
        "ga": my_blue_cmap,
        "qd": my_green_cmap,
        "random-search": my_purple_cmap,
        "full-search": my_red_cmap,
    }
    algorithms = ["random-search", "full-search", "qd", "ga"]

    for target in targets:
        df_dict = {}
        for algorithm in algorithms:
            if algorithm == "ga" or algorithm == "qd":
                filenum_str = "{:0>8}".format(generation - 1)
                file_path = f"../{algorithm}/results/{target}/archives/{filenum_str}.csv"
            elif algorithm == "full-search" or algorithm == "random-search":
                file_path = f"../{algorithm}/results/{target}/archive.csv"
            df_dict[algorithm] = pd.read_csv(file_path)

        dir = f"results/gene_space/{target}"
        os.makedirs(dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        for algorithm in algorithms:
            plt.scatter(
                df_dict[algorithm]["rho"],
                df_dict[algorithm]["nu"],
                c=df_dict[algorithm]["distance"],
                cmap=cmap_dict[algorithm],
                s=5,
                vmin=0,
                vmax=5,
            )
            plt.colorbar()

        plt.xlabel("ρ")
        plt.ylabel("ν")
        plt.tight_layout()
        plt.savefig(f"{dir}/rho_nu.png", dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        for algorithm in algorithms:
            plt.scatter(
                df_dict[algorithm]["recentness"],
                df_dict[algorithm]["frequency"],
                c=df_dict[algorithm]["distance"],
                cmap=cmap_dict[algorithm],
                s=5,
                vmin=0,
                vmax=5,
            )
            plt.colorbar()

        plt.xlabel("recentness")
        plt.ylabel("friendship")
        plt.tight_layout()
        plt.savefig(f"{dir}/r_f.png", dpi=300)
        plt.close()
