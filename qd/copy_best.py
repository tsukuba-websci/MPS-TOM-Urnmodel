import argparse
import os
import shutil

import pandas as pd

if __name__ == "__main__":
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", action="store_true", help="Overwrite destination directory")
    args = parser.parse_args()

    hyperparams_dir = "results/hyperparams-search/"
    targets = ["aps", "twitter"]  # 各targetのリスト

    for target in targets:
        target_dir = os.path.join(hyperparams_dir, target)

        # targetディレクトリ内のファイルを走査
        best_file = None
        best_distance = float("inf")

        for cells in os.listdir(target_dir):
            cells_dir = os.path.join(target_dir, cells)
            if not os.path.isdir(cells_dir):
                continue
            for dim in os.listdir(cells_dir):
                dim_dir = os.path.join(cells_dir, dim)
                file_path = os.path.join(dim_dir, "best.csv")
                print(file_path)
                # best.csvを読み込む
                df = pd.read_csv(file_path)
                distance = df["distance"].iloc[0]
                print(distance)

            if distance < best_distance:
                best_dir = dim_dir
                best_distance = distance

        # 最良の結果をコピーする
        destination_dir = os.path.join("results/", target)
        os.makedirs(destination_dir, exist_ok=True)
        if args.f:
            shutil.copytree(best_dir, destination_dir, dirs_exist_ok=True)
        else:
            shutil.copytree(best_dir, destination_dir)
