from argparse import ArgumentParser
from typing import Dict, cast

import pandas as pd
from history2bd.main import History2BD

from lib.history2vec import History2VecResult
from lib.julia_initializer import JuliaInitializer
from qd import QualityDiversitySearch

if __name__ == "__main__":
    # setup args
    arg_parser = ArgumentParser(description="特定のターゲットに対してQDを使ってモデルをフィッティングする。../data/<target>.csvに所定のファイルが必要。")
    arg_parser.add_argument(
        "target_name",
        type=str,
        choices=["twitter", "aps", "synthetic"],
        help="ターゲットデータ",
    )
    arg_parser.add_argument("rho", type=int, nargs="?", default=None, help="rho")
    arg_parser.add_argument("nu", type=int, nargs="?", default=None, help="nu")
    arg_parser.add_argument("s", type=str, nargs="?", default=None, choices=["SSW", "WSW"], help="strategy")
    args = arg_parser.parse_args()

    target_name: str = args.target_name

    # read target data
    if target_name == "synthetic":
        rho = args.rho
        nu = args.nu
        s = args.s
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
        target_name = f"synthetic/rho{rho}_nu{nu}_s{s}"
        num_generations = 100
    else:
        target_csv = f"../data/{target_name}.csv"
        df = cast(Dict[str, float], pd.read_csv(target_csv).iloc[0].to_dict())
        target = History2VecResult(**df)
        num_generations = 500

    # Set Up Julia
    jl_main, thread_num = JuliaInitializer().initialize()

    # run QD
    for cells in [250, 500, 750]:
        for dim in [64, 128, 256]:
            # load models about the axes of QD
            history2bd = History2BD(
                graph2vec_model_path=f"./models/dim{dim}/graph2vec.pkl",
                standardize_model_path=f"./models/dim{dim}/standardize.pkl",
            )

            qds = QualityDiversitySearch(
                target_name=target_name,
                target=target,
                history2bd=history2bd,
                iteration_num=num_generations,
                thread_num=thread_num,
                jl_main=jl_main,
                dim=dim,
                cells=cells,
            )
            print(f"start {qds.result_dir_path}")
            qds.run()
