import argparse
import csv
import logging
import os

from ga import GA
from history2vec import History2VecResult
from io_utils import dump_json, parse_args, validate
from julia_initializer import JuliaInitializer


def main():
    """実行時にターゲットデータを読み込み，それに対して最も適応度の高いパラメータを遺伝的アルゴリズムで探索する．"""
    # parse arguments
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    population_size, rate, cross_rate = (
        args.population_size,
        args.rate,
        args.cross_rate,
    )
    validate(population_size, rate, cross_rate)

    target_data = args.target_data

    # Set Up Julia
    jl_main, thread_num = JuliaInitializer().initialize()

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=f"log/{target_data}_rate_{rate}_population_{population_size}_cross_rate_{cross_rate}.log",
    )
    logging.info(f"Start GA with population_size={population_size}, rate={rate}, cross_rate={cross_rate}")

    # read target data
    fp = f"../data/{target_data}.csv"
    reader = csv.reader(open(fp, "r"))
    _ = next(reader)
    result = []
    for row in reader:
        if len(row) < 10:
            raise ValueError("Invalid target data.")
        target = History2VecResult(
            gamma=float(row[-10]),
            no=float(row[-9]),
            nc=float(row[-8]),
            oo=float(row[-7]),
            oc=float(row[-6]),
            c=float(row[-5]),
            y=float(row[-4]),
            g=float(row[-3]),
            r=float(row[-2]),
            h=float(row[-1]),
        )

        ga = GA(
            target=target,
            population_size=population_size,
            rate=rate,
            cross_rate=cross_rate,
            history=[],
            jl_main=jl_main,
            thread_num=thread_num,
        )

        min_fitness, target_vec, params, ten_metrics = ga.run()
        result.append((min_fitness, target_vec, params, ten_metrics))

    # sort by fitness
    result = sorted(result, key=lambda x: x[0])
    best_result = result[0]
    if args.prod:
        dir_name = "./log"
        next_idx = 1
        for f in os.listdir(dir_name):
            if f.startswith(target_data) and f.endswith(".json"):
                next_idx += 1
        fp = f"./log/{target_data}_{next_idx}.json"
        dump_json(best_result, fp)
    else:
        os.makedirs(f"./results/grid_search_{target_data}/rate_{rate}", exist_ok=True)
        fp = f"./results/grid_search_{target_data}/rate_{rate}/population{population_size}_cross_rate{cross_rate}.json"
        dump_json(best_result, fp)

    logging.info(f"Finihsed GA. Result is dumped to {fp}")


if __name__ == "__main__":
    main()
