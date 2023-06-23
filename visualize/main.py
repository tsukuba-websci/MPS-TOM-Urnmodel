import argparse

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from visualize.graphs.bar_graph import plot_bar_graph
from visualize.graphs.radar_chart import plot_radar_chart
from visualize.graphs.timeline import plot_timeline


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("graph_type", type=str, choices=["bar", "radar", "timeline"], help="グラフの種類")
    parser.add_argument("target_type", type=str, choices=["empirical", "synthetic"], help="データの種類")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    graph_type = args.graph_type
    target_type = args.target_type

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
        synthetic = pd.read_csv("../data/synthetic_target.csv").set_index(["rho", "nu", "s"]).sort_index()
        synthetic_mean = synthetic.groupby(["rho", "nu", "s"]).mean()

    fm: matplotlib.font_manager.FontManager = matplotlib.font_manager.fontManager
    fm.addfont("./STIXTwoText.ttf")
    plt.rcParams["font.family"] = "STIX Two Text"
    plt.rcParams["font.size"] = 16

    my_color = {
        "red": "#FC8484",
        "dark_red": "#FA5050",
        "light_blue": "#9CC3DA",
        "light_green": "#9CDAA0",
        "dark_blue": "#76ABCB",
        "dark_green": "#51BD56",
        "black": "#505050",
        "orange": "#ff7f0e",
        "blue": "#1f77b4",
        "green": "#9CDAA0",
    }

    if graph_type == "bar":
        plot_bar_graph(target_type, targets, my_color)
    elif graph_type == "radar":
        plot_radar_chart(target_type, targets, my_color)
    elif graph_type == "timeline":
        plot_timeline(target_type, targets, my_color)
