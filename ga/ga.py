import csv
import logging
from typing import Any, List, Tuple

import numpy as np

from lib.history2vec import History2Vec, History2VecResult
from lib.run_model import Params, run_model


class GA:
    target: History2VecResult

    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        cross_rate: float,
        target: History2VecResult,
        target_data: str,
        num_generations: int,
        jl_main: Any,
        thread_num: int,
        archive_dir: str,
        min_val: float = -1.0,
        max_val: float = 1.0,
        debug: bool = True,
        is_grid_search: bool = False,
    ) -> None:
        self.population_size = population_size
        self.min_val = min_val
        self.max_val = max_val
        self.mutation_rate = mutation_rate
        self.cross_rate = cross_rate
        self.num_generations = num_generations

        self.target = target
        self.target_data = target_data
        self.jl_main = jl_main
        self.thread_num = thread_num
        self.histories = [[] for _ in range(self.population_size)]
        self.archives_dir = archive_dir
        self.debug = debug
        self.is_grid_search = is_grid_search

    def tovec(self, history: List[Tuple[int, int]], interval_num: int) -> History2VecResult:
        """相互やり取りの履歴を10個の指標に変換する．

        Args:
            history (List[Tuple[int, int]]): 相互作用履歴
            interval_num (int): 区間数

        Returns:
            History2VecResult: 履歴ベクトル
        """
        return History2Vec(self.jl_main, self.thread_num).history2vec(history, interval_num)

    def fitness_function(self, history: list) -> float:
        """適応度計算．目的関数 * -1 を返す．

        Args:
            history (list): 履歴ベクトル

        Returns:
            float: 適応度
        """
        return -1 * self.objective_function(history)

    def objective_function(self, history: list) -> float:
        """目的関数．ターゲットとの差の絶対値の和を返す．

        Args:
            history (list): 履歴ベクトル

        Returns:
            float: 目的関数の値
        """
        return np.sum(np.abs(np.array(history) - np.array(self.target)))

    def selection(self, population: list, fitness: list) -> list:
        """ルーレット選択．適応度に比例した確率で個体を選択し，親個体にする．この親個体を用いて交叉を行う．

        Args:
            population (list): 各個体のパラメータ (rho, nu, recentness, frequency) のリスト
            fitness (list): 各個体の適応度

        Returns:
            list: 親個体のリスト．交叉を行えるように parents1 (list), parents2 (list) の2つのリストを返す
        """
        parents1 = np.zeros((self.population_size, 4))
        parents2 = np.zeros((self.population_size, 4))
        fitness = np.array(fitness)
        weights = calc_weights(fitness)
        for i in range(self.population_size):
            parents1[i] = population[np.random.choice(self.population_size, p=weights)]
            parents2[i] = population[np.random.choice(self.population_size, p=weights)]
        return parents1, parents2

    def crossover(self, parents1: list, parents2: list, children: list) -> list:
        """交叉．親のうちランダムに選んだものを交叉させる．

        Args:
            parents1 (list): 親1 (rho, nu, recentness, frequency) のリスト
            parents2 (list): 親2 (rho, nu, recentness, frequency) のリスト
            children (list): 子のリスト

        Returns:
            children (list): 子のリスト
        """
        assert len(children) == self.population_size
        for i in range(self.population_size):
            if np.random.rand() < self.cross_rate:
                idx = np.random.randint(4)
                children[i] = np.concatenate([parents1[i][:idx], parents2[i][idx:]])
        return children

    def mutation(self, children: list) -> list:
        """突然変異．子のうちランダムに選んだものを突然変異させる．

        Args:
            children (list): 子のリスト

        Returns:
            children (list): 子のリスト
        """
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                idx = np.random.randint(4)
                children[i][idx] = np.random.uniform(low=self.min_val, high=self.max_val)
        return children

    def dump_population(self, population: list, generation: int, fitness: list) -> None:
        """個体群をファイルに出力する．

        Args:
            population (list): 個体群
            generation (int): 世代数
            fitness (list): 適応度
        """
        fp = f"{self.archives_dir}/{str(generation).zfill(8)}.csv"
        with open(fp, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["rho", "nu", "recentness", "frequency", "objective"])
            for individual, fit in zip(population, fitness):
                writer.writerow([individual[0], individual[1], individual[2], individual[3], -1 * fit])

    def plot():
        return

    def run_init(self) -> list:
        """GAの初期個体群を生成する．

        Returns:
            population (list): 初期個体群 (rho, nu, recentness, frequency) のリスト
        """
        rho = np.random.uniform(low=0, high=30, size=self.population_size)
        nu = np.random.uniform(low=0, high=30, size=self.population_size)
        recentness = np.random.uniform(low=self.min_val, high=self.max_val, size=self.population_size)
        frequency = np.random.uniform(low=self.min_val, high=self.max_val, size=self.population_size)
        population = np.array([rho, nu, recentness, frequency]).T
        return population

    def run(self) -> Tuple[float, History2VecResult, list]:
        """GAの実行を行う．
        Args:
            None
        Returns:
            best_fitness (float): 最良の適応度
            best_history_vec (History2VecResult): 最良の履歴ベクトル
            best_params (list): 最良のパラメータ

        実際には以下のような流れで進化する．
        1. 初期化
        2. 世代ごとに進化
        2.1. やり取りを行う履歴を生成
        3. 適応度計算
        4. 選択
        5. 交叉
        6. 突然変異
        7. 次世代へ
        8. 結果の表示
        """
        population = self.run_init()
        # 世代ごとに進化
        for generation in range(self.num_generations):
            fitness = np.zeros(self.population_size)

            # やり取りを行う履歴を生成し，適応度計算を行う
            for i in range(self.population_size):
                params = Params(
                    rho=population[i][0],
                    nu=population[i][1],
                    recentness=population[i][2],
                    frequency=population[i][3],
                    steps=20000,
                )
                self.histories[i] = run_model(params)
                fitness[i] = self.fitness_function(self.tovec(self.histories[i], 1000))

            # 選択
            parents1, parents2 = self.selection(population, fitness)

            # 親の遺伝子をそのまま子に流しておく
            children = np.zeros((self.population_size, 4))
            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    children[i] = parents1[i]
                else:
                    children[i] = parents2[i]

            # 交叉
            children = self.crossover(parents1, parents2, children)

            # 突然変異
            children = self.mutation(children)

            # 次世代へ
            population = children.copy()

            # 結果の表示
            if self.debug:
                arg = np.argmax(fitness)
                best_fitness = -1 * np.max(fitness)
                best_params = population[arg]
                metrics = self.tovec(self.histories[arg], 10)
                message = f"Generation {generation}: Best fitness = {best_fitness}, Best params = {best_params}, 10Metrics = {metrics}"
                logging.info(message)

            # 個体群の出力
            self.dump_population(population, generation, fitness)

        # 適応度の最小値，ターゲット，最適解，10個の指標を返す
        arg = np.argmax(fitness)
        return (
            -1 * np.max(fitness),
            self.target,
            population[arg],
            self.tovec(self.histories[arg], 10),
        )


def sigmoid(a: float, x: np.ndarray) -> np.ndarray:
    assert a > 0
    return 1 / (1 + np.exp(-a * x))


def calc_weights(x: np.ndarray) -> np.ndarray:
    """適応度から重みを計算する．
    Args:
        x (np.ndarray): 適応度のリスト

    Returns:
        weights (np.ndarray): 重みのリスト
    """

    x = x - np.mean(x)
    weights = sigmoid(5, x)
    weights /= np.sum(weights)
    return weights
