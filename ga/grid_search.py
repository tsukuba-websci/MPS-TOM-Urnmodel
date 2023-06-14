import argparse
import os
from typing import Any, Dict, cast

import pandas as pd
from io_utils import dump_json

from ga import GA
from lib.history2vec import History2VecResult
from lib.julia_initializer import JuliaInitializer


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """コマンドライン引数のパース．

    Args:
        parser (argparse.ArgumentParser): コマンドライン引数のパーサー

    Returns:
        argparse.Namespace: コマンドライン引数のパース結果
            target_data (str): ターゲットデータ
            force (bool): 既存のファイルを上書きするかどうか
    """
    parser.add_argument("target_data", type=str, choices=["twitter", "aps", "synthetic"], help="ターゲットデータ")
    parser.add_argument("-f", "--force", action="store_true", default=False, help="既存のファイルを上書きする．")
    parser.add_argument("rho", type=int, nargs="?", default=None, help="rho")
    parser.add_argument("nu", type=int, nargs="?", default=None, help="nu")
    parser.add_argument("s", type=str, nargs="?", default=None, choices=["SSW", "WSW"], help="strategy")
    args = parser.parse_args()
    return args


class GridSearch:
    def __init__(
        self, target: History2VecResult, target_data: str, output_dir: str, jl_main: Any, thread_num: int, force: bool
    ):
        """コンストラクタ．
        Args:
            target (History2VecResult): ターゲット
            target_data (str): ターゲットデータ
            output_dir (str): 出力先のディレクトリ
            jl_main (Any): Juliaのmain関数
            thread_num (int): Juliaのスレッド数
            force (bool): 既存のファイルを上書きするかどうか
        """
        self.target = target
        self.target_data = target_data
        self.output_dir = output_dir
        self.jl_main = jl_main
        self.thread_num = thread_num
        self.force = force

    def search(
        self,
        mutation_rate_iter: iter,
        cross_rate_iter: iter,
        population_size_iter: iter,
        num_generations: int,
    ) -> None:
        """GAのパラメータをグリッドサーチする．
        Args:
            target (History2VecResult): ターゲット
            mutation_rate_iter (iter): 突然変異率のイテレータ
            cross_rate_iter (iter): 交叉率のイテレータ
            population_size_iter (iter): 個体数のイテレータ
            num_generations (int): 世代数
        """
        for mutation_rate in mutation_rate_iter:
            for cross_rate in cross_rate_iter:
                for population_size in population_size_iter:
                    output_fp = f"{self.output_dir}/mutation_rate_{mutation_rate}_population_{population_size}_cross_rate_{cross_rate}.json"
                    if os.path.exists(output_fp) and not self.force:
                        print(f"{output_fp} already exists. Skip.")
                        continue
                    result = GA(
                        target=self.target,
                        target_data=self.target_data,
                        num_generations=num_generations,
                        population_size=population_size,
                        mutation_rate=mutation_rate,
                        cross_rate=cross_rate,
                        jl_main=self.jl_main,
                        thread_num=self.thread_num,
                        archive_dir=self.output_dir,
                        is_grid_search=True,
                    ).run()
                    # dump result
                    dump_json(
                        result,
                        output_fp,
                    )


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    target_data = args.target_data
    force = args.force

    # read target data
    if target_data == "synthetic":
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
        target_data = f"synthetic/rho{rho}_nu{nu}_s{s}"

    else:
        target_csv = f"../data/{target_data}.csv"
        df = cast(Dict[str, float], pd.read_csv(target_csv).iloc[0].to_dict())
        target = History2VecResult(**df)

    # Set Up Julia
    jl_main, thread_num = JuliaInitializer().initialize()

    # Set Up GridSearch
    mutation_rate_iter = [0.01, 0.02, 0.03, 0.04, 0.05]
    cross_rate_iter = [0.8, 0.85, 0.9, 0.95]
    population_size_iter = [10, 20, 30, 40, 50]
    num_generations = 100
    output_dir = f"./results/grid_search/{target_data}"
    os.makedirs(output_dir, exist_ok=True)
    gs = GridSearch(target, target_data, output_dir, jl_main, thread_num, force)
    gs.search(mutation_rate_iter, cross_rate_iter, population_size_iter, num_generations)


if __name__ == "__main__":
    main()
