import json
import os
import sys


def main():
    """最適なパラメタだったファイルを探す"""
    best_fname = ""
    best_distance = float("inf")

    target_data = sys.argv[1]
    if target_data == "synthetic":
        rho = sys.argv[2]
        nu = sys.argv[3]
        s = sys.argv[4]
        target_data = f"synthetic/rho{rho}_nu{nu}_s{s}"

    respath = f"./results/grid_search/{target_data}/"

    for file in os.listdir(respath):
        if file.endswith(".json"):
            fname = os.path.join(respath, file)
            with open(fname) as f:
                res = json.load(f)
                if res["min_distance"] < best_distance:
                    best_distance = res["min_distance"]
                    best_fname = fname

    print("Best file: ", best_fname)
    print("distance: ", best_distance)


if __name__ == "__main__":
    main()
