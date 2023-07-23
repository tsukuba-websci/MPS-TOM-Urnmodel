import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# NOTE: MPS-TOM-urnmodel/で用意されているpoetry環境ではpandasのversionの依存関係が合わず、動きません。
# pandas>=1.5.3, Jinja2>=3.0.0, matplotlib, numpyのある環境で実行してください。


def export_latex_table(target_type: str, targets: list, generation: int, my_color: dict) -> None:
    my_red_cmap = LinearSegmentedColormap.from_list("my_red_gradient", colors=["white", my_color["red"]])
    my_green_cmap = LinearSegmentedColormap.from_list("my_green_gradient", ["white", my_color["light_green"]])

    contents = "\\begin{table*}[h]\n"

    n = 5
    for target in targets:
        filenum_str = "{:0>8}".format(generation - 1)
        path = f"../qd/results/{target}/archives/{filenum_str}.csv"
        _df = pd.read_csv(path)

        _df.index = pd.Index(np.arange(len(_df)) + 1)
        _df["i"] = _df.index
        _df = _df.head(n)
        _df = _df[["i", "distance", "rho", "nu", "recentness", "frequency"]]

        l = (
            _df.style.hide()
            .background_gradient(
                vmin=1,
                vmax=20,
                subset=["rho", "nu"],
                axis="columns",
                cmap=my_green_cmap,
            )
            .background_gradient(
                vmin=-1,
                vmax=1,
                subset=["recentness", "frequency"],
                axis="columns",
                cmap=my_red_cmap,
            )
            .format(
                precision=2,
                subset=["rho", "nu", "recentness", "frequency"],
            )
            .format(
                precision=3,
                subset=["distance"],
            )
            .relabel_index(
                labels=[r"$\#$", r"d", r"$\rho$", r"$\nu$", r"$r$", r"$f$"],
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
            rf"\begin{{minipage}}[t]{{{1 / len(targets)}\textwidth}}",
            rf"\caption{{best {n} genes for {target}}}",
            rf"\label{{table:best_genes_for_{target}}}",
            l.removesuffix("\n"),
            r"\end{minipage}\hfill",
        ]
        contents += "\n".join(table)

    contents += "\\end{table*}"

    os.makedirs("./results/table", exist_ok=True)
    with open(f"./results/table/{target_type}.tex", "w") as f:
        f.write(contents)


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
        "red": "#FC8484",
        "light_green": "#9CDAA0",
    }

    export_latex_table(target_type, targets, generation, my_color)
