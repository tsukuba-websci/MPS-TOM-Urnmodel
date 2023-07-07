import csv
import os
from argparse import ArgumentParser, Namespace

import pandas as pd

from random_search import RandomSearch

from lib.history2vec import History2VecResult
from lib.julia_initializer import JuliaInitializer


def parse_args(parser: ArgumentParser) -> Namespace:
    """コマンドライン引数を解析する

    Args:
        parser (ArgumentParser): ArgumentParser オブジェクト

    Returns:
        Namespace: 解析結果

    Examples:
        >>> parser = ArgumentParser()
        >>> args = parse_args(parser)
        >>> args.iterations
        100
        >>> args.target
        'aps'
        >>> args.force
        False
        >>> args.debug
        False
    """
    parser.add_argument("--iterations", help="Number of iterations", type=int, default=100)
    parser.add_argument(
        "--target", help="Target Data", type=str, choices=["aps", "twitter", "mixi", "synthetic"], default="aps"
    )
    parser.add_argument("-f", "--force", help="Force to run", action="store_true")
    parser.add_argument("--debug", help="Debug mode", action="store_true")
    return parser.parse_args()


def main():
    parser = ArgumentParser()
    args = parse_args(parser)

    if args.target == "synthetic":
        fp = "../data/synthetic_target.csv"
        rho = input("rho: ")
        nu = input("nu: ")
        s = input("s: ")

        history2vec_results = pd.read_csv("../data/synthetic_target.csv").groupby(["rho", "nu", "s"]).mean()
        row = history2vec_results.query(f"rho == {rho} and nu == {nu} and s == '{s}'").iloc[0]
        target = History2VecResult(
            gamma=row.gamma,
            no=row.no,
            nc=row.nc,
            oo=row.oo,
            oc=row.oc,
            c=row.c,
            y=row.y,
            g=row.g,
            r=row.r,
            h=row.h,
        )

    else:
        fp = f"../data/{args.target}.csv"
        with open(fp) as f:
            reader = csv.reader(f)
            # read header
            _ = next(reader)

            # read target
            data = next(reader)
            target = History2VecResult(
                gamma=float(data[0]),
                c=float(data[1]),
                oc=float(data[2]),
                oo=float(data[3]),
                nc=float(data[4]),
                no=float(data[5]),
                y=float(data[6]),
                r=float(data[7]),
                h=float(data[8]),
                g=float(data[9]),
            )

    jl_main, thread_num = JuliaInitializer().initialize()
    rs = RandomSearch(args.iterations, jl_main, thread_num, target)
    rs.search()

    # 最適解を出力する
    print(f"best solution: (rho, nu, r, f) = {rs.best_solution}")
    print(f"best objective function value: {rs.best_objective}")

    # アーカイブをダンプする
    os.makedirs("./archive", exist_ok=True)
    if args.target == "synthetic":
        fp = f"./archive/synthetic_rho{rho}_nu{nu}_s{s}.csv"
    else:
        fp = f"./archive/{args.target}.csv"
    with open(fp, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["rho", "nu", "r", "f", "objective"])
        for solution, objective in rs.archive:
            writer.writerow([solution[0], solution[1], solution[2], solution[3], objective])


if __name__ == "__main__":
    main()
