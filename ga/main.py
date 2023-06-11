import argparse
import csv
import logging
import os
from typing import Any

from history2vec import History2VecResult
from io_utils import export_individual, parse_args, validate
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
    archive_dir: str,
) -> list:
    """GAを実行し，最も適応度の高い個体の適応度，履歴ベクトル，パラメータ，10個の指標を返す．

    Args:
        reader (csv.reader): CSVリーダー
        target_data (str): ターゲットデータ
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
            archive_dir=archive_dir,
        )

        min_fitness, target_vec, params, ten_metrics = ga.run()
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
    output_base_dir = f"./results/{target_data}"
    os.makedirs(os.path.join(output_base_dir, "archives"), exist_ok=True)

    # read target data
    fp = f"../data/{target_data}.csv"
    reader = csv.reader(open(fp, "r"))
    _ = next(reader)
    min_distance, _, best_individual, _ = run(
        reader=reader,
        target_data=target_data,
        population_size=population_size,
        mutation_rate=mutation_rate,
        cross_rate=cross_rate,
        jl_main=jl_main,
        thread_num=thread_num,
        archive_dir=os.path.join(output_base_dir, "archives"),
    )

    output_fp = os.path.join(output_base_dir, "best.csv")
    export_individual(min_distance, best_individual, output_fp)

    logging.info(f"Finihsed GA. Result is dumped to {fp}")


if __name__ == "__main__":
    main()
