import os
from multiprocessing import Pool
from typing import List

from tqdm import tqdm

from lib.history2vec import History2Vec, History2VecResult
from lib.julia_initializer import JuliaInitializer
from lib.run_model import Params, run_model


class FullSearch:
    def __init__(self, outfile: str, rhos: List[int], nus: List[int]) -> None:
        self.outfile = outfile
        self.rhos = rhos
        self.nus = nus

    def __write_vecs(self, rho: int, nu: int, recentness: float, frequency: float, vecs: List[History2VecResult]):
        for vec in vecs:
            row = ",".join(map(str, [rho, nu, recentness, frequency] + list(vec)))
            with open(f"results/{self.outfile}.csv", "+a") as f:
                f.write(row + "\n")

    def run(self):
        self.jl_main, self.thread_num = JuliaInitializer().initialize()

        os.makedirs("results", exist_ok=True)
        with open(f"results/{self.outfile}.csv", "w") as f:
            f.write(",".join(["rho", "nu", "recentness", "frequency"] + list(History2VecResult._fields)) + "\n")

        ss = [(1.0, 0.0), (0.5, 0.5)]

        bar = tqdm(total=len(self.rhos) * len(self.nus) * len(ss))
        for rho in self.rhos:
            for nu in self.nus:
                for s in ss:
                    n = 10
                    rhos: List[float] = [rho for _ in range(n)]
                    nus: List[float] = [nu for _ in range(n)]
                    recentnesses: List[float] = [s[0] for _ in range(n)]
                    frequency: List[float] = [s[1] for _ in range(n)]
                    steps = [20000 for _ in range(10)]
                    params_list = map(
                        lambda t: Params(*t),
                        zip(rhos, nus, recentnesses, frequency, steps),
                    )
                    with Pool(self.thread_num) as pool:
                        histories = pool.map(run_model, params_list)
                    vecs = History2Vec(self.jl_main, self.thread_num).history2vec_parallel(histories, 1000)

                    self.__write_vecs(rho, nu, s[0], s[1], vecs)
                    bar.update(1)


if __name__ == "__main__":
    rhos = list(range(1, 20 + 1))
    nus = list(range(1, 20 + 1))
    outfile = "full-search"

    runner = FullSearch(outfile, rhos, nus)
    runner.run()
