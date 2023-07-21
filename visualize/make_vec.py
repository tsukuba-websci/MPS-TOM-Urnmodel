import argparse
import csv
import os
from multiprocessing import Pool
from typing import Dict, cast
import numpy as np

import pandas as pd

from lib.history2vec import History2Vec, History2VecResult
from lib.julia_initializer import JuliaInitializer
from lib.run_model import Params, run_model
from qd.history2bd.main import History2BD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_type", type=str, choices=["empirical", "synthetic"], help="データの種類")
    target_type = parser.parse_args().target_type

    if target_type == "empirical":
        targets = ["mixi"]
        # targets = ["aps", "twitter"]
        # targets = ["aps", "twitter", "mixi"]
    elif target_type == "synthetic":
        targets = [
            f"{target_type}/rho5_nu5_sSSW",
            f"{target_type}/rho5_nu5_sWSW",
            f"{target_type}/rho5_nu15_sSSW",
            f"{target_type}/rho5_nu15_sWSW",
            f"{target_type}/rho20_nu7_sSSW",
            f"{target_type}/rho20_nu7_sWSW",
        ]

    algorithms = ["ga", "qd"]

    jl_main, thread_num = JuliaInitializer().initialize()
    history2vec_ = History2Vec(jl_main, thread_num)
    history2bd = History2BD(
        graph2vec_model_path="../qd/models/dim128/graph2vec.pkl",
        standardize_model_path="../qd/models/dim128/standardize.pkl",
    )

    if target_type == "empirical":
        generation = 500
    else:
        generation = 100

    header = ["vec" + str(i) for i in range(0, 128)]
    header.append("distance")

    for target in targets:
        os.makedirs(f"results/vec/{target}", exist_ok=True)

        # ターゲットデータを読み込む
        target_csv = f"../data/{target}.csv"
        df = cast(Dict[str, float], pd.read_csv(target_csv).iloc[0].to_dict())
        target_vec = History2VecResult(**df)

        for algorithm in algorithms:
            filenum_str = "{:0>8}".format(generation - 1)
            file_path = f"../{algorithm}/results/{target}/archives/{filenum_str}.csv"
            df = pd.read_csv(file_path)

            params_list = []
            for i in range(df.shape[0]):
                row = cast(Dict[str, float], df.iloc[i].to_dict())
                param = Params(row["rho"], row["nu"], row["recentness"], row["frequency"], 20000)
                params_list.append(param)

            with Pool(thread_num) as pool:
                histories = pool.map(run_model, params_list)

            bcs = history2bd.run(histories)
            history_vecs = history2vec_.history2vec_parallel(histories, 1000)
            distances = []
            for history_vec in history_vecs:
                distance: np.float64 = np.sum(np.abs((np.array(history_vec) - np.array(target_vec))))  # type: ignore
                distances.append(distance)

            with open(f"results/vec/{target}/{algorithm}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for bc, distance in zip(bcs, distances):
                    writer.writerow(np.append(bc, distance))
