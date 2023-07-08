import argparse
import os
import shutil

import pandas as pd


def copy_best_results(targets, source_dir, destination_dir, overwrite=False):
    for target in targets:
        target_dir = os.path.join(source_dir, target)
        if not os.path.isdir(target_dir):
            continue

        best_dir = None
        best_distance = float("inf")

        for cells in os.listdir(target_dir):
            cells_dir = os.path.join(target_dir, cells)
            if not os.path.isdir(cells_dir):
                continue

            for dim in os.listdir(cells_dir):
                dim_dir = os.path.join(cells_dir, dim)
                file_path = os.path.join(dim_dir, "best.csv")

                df = pd.read_csv(file_path)
                distance = df["distance"].iloc[0]

                if distance < best_distance:
                    best_dir = dim_dir
                    best_distance = distance

        if best_dir:
            print(f"best_dir: {best_dir}")
            destination_target_dir = os.path.join(destination_dir, target)
            os.makedirs(destination_target_dir, exist_ok=True)
            if overwrite:
                shutil.copytree(best_dir, destination_target_dir, dirs_exist_ok=True)
            else:
                shutil.copytree(best_dir, destination_target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", action="store_true", help="Overwrite destination directory")
    args = parser.parse_args()

    hyperparams_dir = "results/hyperparams-search/"
    destination_dir = "results/"

    # FIXME: 結果が出ていないものはコメントアウトして実行して下さい
    targets = [
        "aps",
        "twitter",
        "synthetic/rho5_nu5_sSSW",
        "synthetic/rho5_nu5_sWSW",
        "synthetic/rho5_nu15_sSSW",
        "synthetic/rho5_nu15_sWSW",
        "synthetic/rho20_nu7_sSSW",
        "synthetic/rho20_nu7_sWSW",
    ]

    copy_best_results(targets, hyperparams_dir, destination_dir, overwrite=args.f)
