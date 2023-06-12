import argparse
import logging
import os
from typing import Any, Dict, cast

import pandas as pd
from history2vec import History2VecResult
from io_utils import export_individual, parse_args, pass_run, validate
from julia_initializer import JuliaInitializer

from ga import GA


def config_logging(target_data: str, mutation_rate: float, population_size: int, cross_rate: float):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=f"log/{target_data}_mutation_rate_{mutation_rate}_population_{population_size}_cross_rate_{cross_rate}.log",
    )
    logging.info(
        f"Start GA with population_size={population_size}, mutation_rate={mutation_rate}, cross_rate={cross_rate}"
    )


def run(
    target: History2VecResult,
    target_data: str,
    num_generations: int,
    population_size: int,
    mutation_rate: float,
    cross_rate: float,
    jl_main: Any,
    thread_num: int,
    archive_dir: str,
) -> list:
    """GAを実行し，最も適応度の高い個体の適応度，履歴ベクトル，パラメータ，10個の指標を返す．

    Args:
        target (History2VecResult): ターゲットの10個の指標
        target_data (str): ターゲットデータ
        num_generations (int): 世代数
        population_size (int): 個体数
        mutation_rate (float): 突然変異率
        cross_rate (float): 交叉率
        jl_main (Any): Juliaのmain関数
        thread_num (int): Juliaのスレッド数
        archive_dir (str): アーカイブの出力先

    Returns:
        list: 最も適応度の高い個体の適応度，履歴ベクトル，パラメータ，10個の指標
    """
    result = []
    logging.info(f"Target Data: {target_data}")
    logging.info(f"Target Metrics: {target}")

    ga = GA(
        target=target,
        target_data=target_data,
        num_generations=num_generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        cross_rate=cross_rate,
        jl_main=jl_main,
        thread_num=thread_num,
        archive_dir=archive_dir,
    )

    min_fitness, target_vec, params, ten_metrics = ga.run()
    logging.info(f"min_fitness={min_fitness}, target_vec={target_vec}, params={params}, ten_metrics={ten_metrics}")
    result.append((min_fitness, target_vec, params, ten_metrics))

    # sort by fitness
    result = sorted(result, key=lambda x: x[0])
    min_distance, target, best_individual, metrics = result[0]
    return min_distance, target, best_individual, metrics


def main():
    """実行時にターゲットデータを読み込み，それに対して最も適応度の高いパラメータを遺伝的アルゴリズムで探索する．"""
    # parse arguments
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    population_size, mutation_rate, cross_rate = (
        args.population_size,
        args.mutation_rate,
        args.cross_rate,
    )
    validate(population_size, mutation_rate, cross_rate)

    target_data = args.target_data

    # read target data
    if target_data == "synthetic":
        os.makedirs(f"./log/{target_data}", exist_ok=True)

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
        num_generations = 100
    else:
        target_csv = f"../data/{target_data}.csv"
        df = cast(Dict[str, float], pd.read_csv(target_csv).iloc[0].to_dict())
        target = History2VecResult(**df)
        num_generations = 500

    # setting output directory
    output_base_dir = f"./results/{target_data}"
    os.makedirs(os.path.join(output_base_dir, "archives"), exist_ok=True)
    output_fp = os.path.join(output_base_dir, "best.csv")

    # configure logging
    config_logging(target_data, mutation_rate, population_size, cross_rate)

    # check if the run is already finished
    if pass_run(args.force, output_fp):
        logging.info("GA is skipped.")
        print("GA search is skipped. Use --force option to run GA.")
        return
    elif args.force:
        print("GA is forced to run. This may overwrite existing result.")
        logging.warning("GA is forced to run.")

    # Set Up Julia
    jl_main, thread_num = JuliaInitializer().initialize()

    min_distance, _, best_individual, _ = run(
        target=target,
        target_data=target_data,
        num_generations=num_generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        cross_rate=cross_rate,
        jl_main=jl_main,
        thread_num=thread_num,
        archive_dir=os.path.join(output_base_dir, "archives"),
    )

    export_individual(min_distance, best_individual, output_fp)

    logging.info(f"Finihsed GA. Result is dumped to {target_data}")


if __name__ == "__main__":
    main()
