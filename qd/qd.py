import os
import pickle
import time
from multiprocessing import Pool
from typing import Any, List, Union

import numpy as np
import pandas as pd
import ribs.emitters as emitters
import ribs.schedulers as schedulers
from history2bd.main import History2BD
from ribs.archives import CVTArchive
from tqdm import tqdm

from lib.history2vec import History2Vec, History2VecResult
from lib.run_model import Params, run_model


class QualityDiversitySearch:
    target_name: str
    target: History2VecResult
    history2bd: History2BD
    thread_num: int = 0
    jl_main: Any
    result_dir_path: str
    archives_dir_path: str
    iteration_num: int

    def __init__(
        self,
        target_name: str,
        target: History2VecResult,
        history2bd: History2BD,
        iteration_num: int,
        thread_num: int,
        jl_main: Any,
        dim: int,
        cells: int,
        result_dir_path: str,
    ) -> None:
        self.target = target
        self.history2bd = history2bd
        self.target_name = target_name
        self.iteration_num = iteration_num
        self.thread_num = thread_num
        self.jl_main = jl_main
        self.dim = dim
        self.cells = cells
        self.result_dir_path = result_dir_path
        self.archives_dir_path = f"{self.result_dir_path}/archives"

        os.makedirs(self.archives_dir_path, exist_ok=True)

    def prepare_archive(self) -> CVTArchive:
        archive: Union[CVTArchive, None] = None
        if os.path.exists(f"{self.result_dir_path}/archive.pkl"):
            with open(f"{self.result_dir_path}/archive.pkl", "rb") as f:
                archive = pickle.load(f)
        else:
            archive = CVTArchive(
                solution_dim=4,
                cells=self.cells,
                ranges=[(-5, 5) for _ in range(self.dim)],
            )
        assert archive is not None, "archive should not be None!"
        return archive

    def save_archive(self, archive: CVTArchive) -> pd.DataFrame:
        # save latest archive
        with open(f"{self.result_dir_path}/archive.pkl", "wb") as file:
            pickle.dump(archive, file)

        # save archive as csv
        df = archive.as_pandas()
        df.rename(
            columns={
                "solution_0": "rho",
                "solution_1": "nu",
                "solution_2": "recentness",
                "solution_3": "frequency",
            },
            inplace=True,
        )
        df["objective"] = -df["objective"]
        df.rename(columns={"objective": "distance"}, inplace=True)
        df = df[["rho", "nu", "recentness", "frequency", "distance"]].sort_values(by="distance", ascending=True)
        df.to_csv(f"{self.archives_dir_path}/{iter:0>8}.csv", index=False)
        return df

    def set_params_list(self, sols: List[np.ndarray]) -> List[Params]:
        rhos: List[float] = list(map(lambda sol: sol[0].item(), sols))
        nus: List[float] = list(map(lambda sol: sol[1].item(), sols))
        recentnesses: List[float] = list(map(lambda sol: sol[2].item(), sols))
        frequency: List[float] = list(map(lambda sol: sol[3].item(), sols))
        steps = [20000 for _ in range(len(rhos))]

        params_list = map(
            lambda t: Params(*t),
            zip(rhos, nus, recentnesses, frequency, steps),
        )
        return params_list

    def print_status(self, archive: CVTArchive, iter: int, start_time: time) -> None:
        elapsed_time = time.time() - start_time
        print(f"> {iter} iters completed after {elapsed_time:.2f} s")
        print(f"  - Archive Size: {len(archive)}")
        assert archive.stats is not None, "archive.stats is None!"
        print(f"  - Max Score: {archive.stats.obj_max}")

    def run(self):
        history2vec_ = History2Vec(self.jl_main, self.thread_num)

        archive: Union[CVTArchive, None] = self.prepare_archive()

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

        for iter in tqdm(range(already, self.iteration_num), desc=self.target_name):
            # Request models from the scheduler
            sols = optimizer.ask()

            params_list = self.set_params_list(sols)

            # Evaluate the models and record the objectives and measuress.
            with Pool(self.thread_num) as pool:
                histories = pool.map(run_model, params_list)

            history_vecs = history2vec_.history2vec_parallel(histories, 1000)

            bcs = self.history2bd.run(histories)

            objs = []
            for history_vec in history_vecs:
                obj: np.float64 = np.sum(np.abs((np.array(history_vec) - np.array(self.target))))  # type: ignore
                objs.append(-obj)

            # Send the results back to the scheduler
            optimizer.tell(objs, bcs)

            # save archive
            df = self.save_archive(archive)

            if iter % 25 == 0:
                self.print_status(archive, iter, start_time)

        # save best result as csv
        if not os.path.exists(f"{self.result_dir_path}/best.csv"):
            df.head(1).to_csv(f"{self.result_dir_path}/best.csv", index=False)
