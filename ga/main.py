import argparse
import csv
import logging
import os
from typing import Any

from history2vec import History2VecResult
from io_utils import parse_args, validate
from julia_initializer import JuliaInitializer

from ga import GA


def run(
    reader: csv.reader,
    target_data: str,
    population_size: int,
    mutation_rate: float,
    cross_rate: float,
    jl_main: Any,
    thread_num: int,
) -> list:
    """GAを実行する．

    Args:
        reader (csv.reader): CSVリーダー
        target_data (str): ターゲットデータ
        population_size (int): 個体数
        mutation_rate (float): 突然変異率
        cross_rate (float): 交叉率
        jl_main (Any): Juliaのmain関数
        thread_num (int): Juliaのスレッド数
    """
    result = []
    for row in reader:
        if len(row) < 10:
            raise ValueError("Invalid target data.")
        target = History2VecResult(
            gamma=float(row[0]),
            c=float(row[1]),
            oc=float(row[2]),
            oo=float(row[3]),
            nc=float(row[4]),
            no=float(row[5]),
            y=float(row[6]),
            r=float(row[7]),
            h=float(row[8]),
            g=float(row[9]),
        )

        ga = GA(
            target=target,
            target_data=target_data,
            population_size=population_size,
            mutation_rate=mutation_rate,
            cross_rate=cross_rate,
            jl_main=jl_main,
            thread_num=thread_num,
        )

        min_fitness, target_vec, params, ten_metrics = ga.run()
        result.append((min_fitness, target_vec, params, ten_metrics))

    # sort by fitness
    result = sorted(result, key=lambda x: x[0])

    return result


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

    # Set Up Julia
    jl_main, thread_num = JuliaInitializer().initialize()

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=f"log/{target_data}_mutation_rate_{mutation_rate}_population_{population_size}_cross_rate_{cross_rate}.log",
    )
    logging.info(
        f"Start GA with population_size={population_size}, mutation_rate={mutation_rate}, cross_rate={cross_rate}"
    )

    # setting output directory
    output_dir = f"./results/{target_data}/"
    os.makedirs(output_dir, exist_ok=True)

    # read target data
    fp = f"../data/{target_data}.csv"
    reader = csv.reader(open(fp, "r"))
    _ = next(reader)
    result = run(
        reader=reader,
        target_data=target_data,
        population_size=population_size,
        mutation_rate=mutation_rate,
        cross_rate=cross_rate,
        jl_main=jl_main,
        thread_num=thread_num,
    )

    # TODO: 出力形式，出力先を変更する

    logging.info(f"Finihsed GA. Result is dumped to {fp}")


if __name__ == "__main__":
    main()
