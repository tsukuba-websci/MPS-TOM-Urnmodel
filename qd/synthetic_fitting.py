from dataclasses import dataclass
from typing import List, Literal

import pandas as pd
from history2bd.main import History2BD
from main import QualityDiversitySearch

from lib.history2vec import History2VecResult
from lib.run_model import Params

Strategy = Literal["SSW", "WSW"]


@dataclass
class Params:
    rho: int
    nu: int
    s: Strategy


def run(rhos: List[int], nus: List[int]):
    history2bd = History2BD(
        graph2vec_model_path="./models/graph2vec.pkl",
        standardize_model_path="./models/standardize.pkl",
    )

    history2vec_results = pd.read_csv("../data/synthetic_target.csv").groupby(["rho", "nu", "s"]).mean()

    strategies: List[Strategy] = ["SSW", "WSW"]

    params_list: List[Params] = []

    for rho in rhos:
        for nu in nus:
            for s in strategies:
                params_list.append(Params(rho, nu, s))

    for params in params_list:
        row = history2vec_results.query(f"rho == {params.rho} and nu == {params.nu} and s == '{params.s}'").iloc[0]
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

        qds = QualityDiversitySearch(
            task_id=f"synthetic/rho{params.rho}_nu{params.nu}_s{params.s}",
            target=target,
            history2bd=history2bd,
            iteration_num=100,
        )
        qds.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument("rhos")
    arg_parser.add_argument("nus")

    args = arg_parser.parse_args()

    rhos_str: str = args.rhos
    rhos: List[int] = list(map(int, rhos_str.split(",")))

    nus_str: str = args.nus
    nus: List[int] = list(map(int, nus_str.split(",")))

    run(rhos, nus)
