import csv
import os
from typing import Any, List, Tuple

import numpy as np

from lib.history2vec import History2Vec, History2VecResult
from lib.julia_initializer import JuliaInitializer
from lib.run_model import Params, run_model


class RandomSearch:
    target: History2VecResult

    def __init__(
        self,
        num_generations: int,
        jl_main: Any,
        thread_num: int,
    ) -> None:
        self.num_generations = num_generations
        self.jl_main = jl_main
        self.thread_num = thread_num

    def tovec(self, history: List[Tuple[int, int]], interval_num: int) -> History2VecResult:
        """相互やり取りの履歴を10個の指標に変換する．
        Args:
            history (List[Tuple[int, int]]): 相互作用履歴
            interval_num (int): 区間数
        Returns:
            History2VecResult: 履歴ベクトル
        """
        return History2Vec(self.jl_main, self.thread_num).history2vec(history, interval_num)

    def make_new_solution(self) -> tuple:
        """新しい解をランダムに生成する．

        Returns:
            tuple: 遺伝子を表す　(rho, nu, r, f) のタプル
        """
        rho = np.random.randint(0, 30)
        nu = np.random.randint(0, 30)
        # r, f は -1 から 1 の間の実数
        r = np.random.rand() * 2 - 1
        f = np.random.rand() * 2 - 1
        solution = (rho, nu, r, f)
        if not self._validate(solution):
            raise ValueError("invalid solution was generated")
        return solution

    def _validate(self, solution: tuple) -> bool:
        """解が探索範囲を満たしているかを判定する．

        Args:
            solution (tuple): 遺伝子を表す　(rho, nu, r, f) のタプル

        Returns:
            bool: 探索範囲を満たしている場合は True そうでない場合は False
        """
        rho, nu, r, f = solution
        if rho < 0 or rho > 30:
            return False
        if nu < 0 or nu > 30:
            return False
        if r < -1 or r > 1:
            return False
        if f < -1 or f > 1:
            return False
        return True

    def run(self) -> None:
        for _ in range(self.num_generations):
            solution = self.make_new_solution()
            histries = run_model(
                Params(
                    rho=solution[0],
                    nu=solution[1],
                    recentness=solution[2],
                    frequency=solution[3],
                    steps=20000,
                )
            )
            vec = self.tovec(histries, 1000)


if __name__ == "__main__":
    iterations = 500

    jl_main, thread_num = JuliaInitializer().initialize()
    rs = RandomSearch(iterations, jl_main, thread_num)
    rs.run()
