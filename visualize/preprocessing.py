import csv
import os
from typing import Dict, cast

import pandas as pd

from lib.history2vec import History2Vec
from lib.julia_initializer import JuliaInitializer
from lib.run_model import Params, run_model

targets = ["aps", "twitter"]
algorithms = ["ga", "qd"]


if __name__ == "__main__":
    jl_main, thread_num = JuliaInitializer().initialize()
    history2vec_ = History2Vec(jl_main, thread_num)
    for target in targets:
        os.makedirs(f"results/fitted/{target}", exist_ok=True)
        for algorithm in algorithms:
            path = f"../{algorithm}/results/{target}/best.csv"
            df = pd.read_csv(path)
            df = cast(Dict[str, float], pd.read_csv(path).iloc[0].to_dict())
            param = Params(df["rho"], df["nu"], df["recentness"], df["frequency"], 20000)
            header = ["gamma", "no", "nc", "oo", "oc", "c", "y", "g", "r", "h"]
            res = []
            for _ in range(10):
                history = run_model(param)
                history2vec_result = history2vec_.history2vec(history, 1000)
                res.append(history2vec_result)

            with open(f"results/fitted/{target}/{algorithm}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in res:
                    writer.writerow(row)
