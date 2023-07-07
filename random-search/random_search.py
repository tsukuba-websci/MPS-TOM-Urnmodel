from typing import Any, List, Tuple

import numpy as np

from lib.history2vec import History2Vec, History2VecResult
from lib.run_model import Params, run_model


class RandomSearch:
    target: History2VecResult

    def __init__(
        self,
        num_generations: int,
        jl_main: Any,
        thread_num: int,
        target: History2VecResult = None,
    ) -> None:
        self.target = target
        self.num_generations = num_generations
        self.jl_main = jl_main
        self.thread_num = thread_num
        self.best_solution = (-1, -1, -1, -1)
        self.best_objective = float("inf")
        self.archive = []

    def tovec(self, history: List[Tuple[int, int]], interval_num: int) -> History2VecResult:
        """相互やり取りの履歴を10個の指標に変換する．
        Args:
            history (List[Tuple[int, int]]): 相互作用履歴
            interval_num (int): 区間数
        Returns:
            History2VecResult: 履歴ベクトル
        """
        return History2Vec(self.jl_main, self.thread_num).history2vec(history, interval_num)

    def objective_function(self, history: list) -> float:
        """目的関数．ターゲットとの差の絶対値の和を返す．

        Args:
            history (list): 履歴ベクトル

        Returns:
            float: 目的関数の値
        """
        return np.sum(np.abs(np.array(history) - np.array(self.target)))

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

    def search(self) -> None:
        """ランダムサーチによる遺伝子探索を行う．

        具体的には一定回数以下の手順を行うことで探索を行う．

        1. 新しい解を生成する．
        2. 満たしている場合は目的関数を計算する．（ただし目的関数は，`ターゲットデータを表す10個の指標` と `解を表す10個の指標` の差の絶対値の和を計算する）
        3. 目的関数の値が最小値を更新している場合は最小値を更新する．
        4. 1 に戻る．
        """

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
            objective = self.objective_function(self.tovec(histries, 1000))
            if objective < self.best_objective:
                self.best_solution = solution
                self.best_objective = objective
            self.archive.append((solution, objective))
