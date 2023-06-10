import os
import pickle
import time
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Any, Dict, List, Union, cast

import numpy as np
import pandas as pd
import ribs.emitters as emitters
import ribs.schedulers as schedulers
from history2bd.main import History2BD
from ribs.archives import CVTArchive
from tqdm import tqdm

from lib.history2vec import History2Vec, History2VecResult
from lib.julia_initializer import JuliaInitializer
from lib.run_model import Params, run_model


class QualityDiversitySearch:
    task_name: str
    target: History2VecResult
    history2bd: History2BD
    thread_num: int = 0
    jl_main: Any
    archive_file_path: str
    archives_dir_path: str
    iteration_num: int

    def __init__(
        self,
        task_id: str,
        target: History2VecResult,
        history2bd: History2BD,
        iteration_num: int = 500,
    ) -> None:
        self.target = target
        self.history2bd = history2bd
        self.task_name = task_id
        self.iteration_num = iteration_num

        result_dir = f"results/{self.task_name}"

        self.archive_file_path = f"{result_dir}/archive.pkl"
        self.archives_dir_path = f"{result_dir}/archives"

        os.makedirs(self.archives_dir_path, exist_ok=True)

    def run(self):
        self.jl_main, self.thread_num = JuliaInitializer().initialize()

        archive: Union[CVTArchive, None] = None
        if os.path.exists(self.archive_file_path):
            with open(self.archive_file_path, "rb") as f:
                archive = pickle.load(f)
        else:
            archive = CVTArchive(
                solution_dim=4,
                cells=500,
                ranges=[(-5, 5) for _ in range(128)],
            )
        assert archive is not None, "archive should not be None!"

        already = 0
        if os.path.exists(self.archives_dir_path):
            already = len(os.listdir(self.archives_dir_path))

        initial_model = np.zeros(4)
        bounds = [
            (0, 30),  # 1 <= rho <= 20
            (0, 30),  # 1 <= nu <= 20
            (-1, 1),  # -1 <= recentness <= 1
            (-1, 1),  # -1 <= frequency <= 1
        ]
        emitters_ = [
            emitters.EvolutionStrategyEmitter(
                archive=archive,
                x0=initial_model,
                sigma0=1.0,
                bounds=bounds,
                ranker="2imp",
            )
            for _ in range(5)
        ]
        optimizer = schedulers.Scheduler(archive, emitters_)

        start_time = time.time()

        for iter in tqdm(range(already, self.iteration_num), desc=self.task_name):
            # Request models from the scheduler
            sols = optimizer.ask()

            rhos: List[float] = list(map(lambda sol: sol[0].item(), sols))
            nus: List[float] = list(map(lambda sol: sol[1].item(), sols))
            recentnesses: List[float] = list(map(lambda sol: sol[2].item(), sols))
            friendships: List[float] = list(map(lambda sol: sol[3].item(), sols))
            steps = [20000 for _ in range(len(rhos))]

            objs = []
            params_list = map(
                lambda t: Params(*t),
                zip(rhos, nus, recentnesses, friendships, steps),
            )

            # Evaluate the models and record the objectives and measuress.
            with Pool(self.thread_num) as pool:
                histories = pool.map(run_model, params_list)

            history_vecs = History2Vec(self.jl_main, self.thread_num).history2vec_parallel(histories, 1000)

            bcs = self.history2bd.run(histories)

            for history_vec in history_vecs:
                obj: np.float64 = np.sum(np.abs((np.array(history_vec) - np.array(self.target))))  # type: ignore
                objs.append(-obj)

            # Send the results back to the scheduler
            optimizer.tell(objs, bcs)

            # save latest archive
            with open(self.archive_file_path, "wb") as file:
                pickle.dump(archive, file)

            # T0DO: csv形式で結果を出力する

            if iter % 25 == 0:
                elapsed_time = time.time() - start_time
                print(f"> {iter} iters completed after {elapsed_time:.2f} s")
                print(f"  - Archive Size: {len(archive)}")
                assert archive.stats is not None, "archive.stats is None!"
                print(f"  - Max Score: {archive.stats.obj_max}")


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="特定のターゲットに対してQDを使ってモデルをフィッティングする。data/<task_id>.csvに所定のファイルが必要。")
    arg_parser.add_argument("task_id", type=str, help="タスクを識別するための名前")
    args = arg_parser.parse_args()

    task_id: str = args.task_id
    target_csv: str = f"../data/{task_id}.csv"

    history2bd = History2BD(
        graph2vec_model_path="./models/graph2vec.pkl",
        standardize_model_path="./models/standardize.pkl",
    )

    df = cast(Dict[str, float], pd.read_csv(target_csv).iloc[0].to_dict())
    target = History2VecResult(**df)

    qds = QualityDiversitySearch(
        task_id=task_id,
        target=target,
        history2bd=history2bd,
    )

    qds.run()
