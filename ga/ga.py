import argparse
import copy
import csv
import os
from typing import Any, List, Tuple
import logging

import numpy as np

from history2vec import History2Vec, History2VecResult, Params
from io_utils import dump_json, parse_args, validate
from julia_initializer import JuliaInitializer
from run_model import run_model


class GA:
    target: History2VecResult

    def __init__(
        self,
        population_size: int,
        rate: float,
        cross_rate: float,
        target: History2VecResult,
        target_data: str,
        history: list,
        jl_main: Any,
        thread_num: int,
        min_val: float = -1.0,
        max_val: float = 1.0,
        debug: bool = True,
        is_grid_search: bool = False,
    ) -> None:
        self.population_size = population_size
        self.min_val = min_val
        self.max_val = max_val
        self.rate = rate
        self.cross_rate = cross_rate
        # FIXME: 一旦デバッグのために10にする
        self.num_generations = 10
        # self.num_generations = 500

        self.target = target
        self.target_data = target_data
        self.history = history
        self.jl_main = jl_main
        self.thread_num = thread_num
        self.history_vec = []
        self.archives_fp = f"./results/{target_data}/{str(len(os.listdir(f'./results/{target_data}'))).zfill(3)}"
        self.debug = debug
        self.is_grid_search = is_grid_search

    def tovec(self, history: List[Tuple[int, int]], interval_num: int) -> History2VecResult:
        """履歴をベクトルに変換する．実際にはHistry2vec.history2vecを呼び出すラッパーとして振る舞っている

        Args:
            history (List[Tuple[int, int]]): 相互作用履歴
            interval_num (int): 区間数

        Returns:
            History2VecResult: 履歴ベクトル
        """
        return History2Vec(self.jl_main, self.thread_num).history2vec(history, interval_num)

    def fitness_function(self, history_vec: list) -> float:
        """適応度計算．とりあえず，目的関数 * -1 を返す．

        Args:
            history_vec (list): 履歴ベクトル

        Returns:
            float: 適応度
        """
        return -1 * self.objective_function(history_vec)

    def objective_function(self, history_vec: list) -> float:
        """目的関数．ターゲットとの差の絶対値の和を返す．

        Args:
            history_vec (list): 履歴ベクトル

        Returns:
            float: 目的関数の値
        """
        return np.sum(np.abs(np.array(history_vec) - np.array(self.target)))

    def crossover(self, parents1: list, parents2: list) -> list:
        """交叉．親のうちランダムに選んだものを交叉させる．

        Args:
            parents1 (list): 親1 (rho, nu, recentness, friendship) のリスト
            parents2 (list): 親2 (rho, nu, recentness, friendship) のリスト

        Returns:
            child (list): 子のリスト (rho, nu, recentness, friendship)
        """
        child = np.zeros(4)
        idx = np.random.randint(4)
        child[:idx] = parents1[:idx]
        child[idx:] = parents2[idx:]
        return child

    def mutation(self, child: list) -> list:
        """突然変異．子のうちランダムに選んだものを突然変異させる．

        Args:
            child (list): 子のリスト

        Returns:
            child (list): 子のリスト
        """
        idx = np.random.randint(4)
        child[idx] = np.random.uniform(low=self.min_val, high=self.max_val)
        return child

    def dump_population(self, population: list, generation: int, fitness: list) -> None:
        """個体群をファイルに出力する．

        Args:
            population (list): 個体群
            generation (int): 世代数
            fitness (list): 適応度
        """
        os.makedirs(f"{self.archives_fp}/archives", exist_ok=True)
        fp = f"{self.archives_fp}/archives/{str(generation).zfill(8)}.csv"
        with open(fp, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["rho", "nu", "recentness", "friendship", "fitness"])
            for individual, fit in zip(population, fitness):
                writer.writerow([individual[0], individual[1], individual[2], individual[3], fit])

    def plot():
        return

    def run_init(self) -> list:
        """GAの初期個体群を生成する．

        Returns:
            population (list): 初期個体群 (rho, nu, recentness, friendship) のリスト
        """
        rho = np.random.uniform(low=0, high=30, size=self.population_size)
        nu = np.random.uniform(low=0, high=30, size=self.population_size)
        recentness = np.random.uniform(low=self.min_val, high=self.max_val, size=self.population_size)
        friendship = np.random.uniform(low=self.min_val, high=self.max_val, size=self.population_size)
        population = np.array([rho, nu, recentness, friendship]).T
        return population

    def run(self) -> Tuple[float, History2VecResult, list]:
        """GAの実行．
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
        # ディレクトリの作成
        os.makedirs(self.archives_fp, exist_ok=True)
        population = self.run_init()
        # 世代ごとに進化
        for generation in range(1, self.num_generations + 1):
            fitness = np.zeros(self.population_size)

            # やり取りを行う履歴を生成し，適応度計算を行う
            for i in range(self.population_size):
                params = Params(
                    rho=population[i][0],
                    nu=population[i][1],
                    recentness=population[i][2],
                    friendship=population[i][3],
                    steps=100,
                )
                self.history = run_model(params)
                fitness[i] = self.fitness_function(self.tovec(self.history, 10))

            # 選択
            parents1 = np.zeros((self.population_size, 4))
            parents2 = np.zeros((self.population_size, 4))
            for i in range(self.population_size):
                weights = 1 / fitness
                weights /= np.sum(weights)
                parents1[i] = population[np.random.choice(self.population_size, p=weights)]
                parents2[i] = population[np.random.choice(self.population_size, p=weights)]

            # 最良の個体を残す
            tmp_max_arg = np.argmax(fitness)
            tmp_max_individual = copy.deepcopy(population[tmp_max_arg])

            # 親の遺伝子をそのまま子に流しておく
            children = np.zeros((self.population_size, 4))
            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    children[i] = parents1[i]
                else:
                    children[i] = parents2[i]

            # 交叉
            for i in range(self.population_size):
                if np.random.rand() < self.cross_rate:
                    children[i] = self.crossover(parents1[i], parents2[i])

            # 突然変異
            for i in range(self.population_size):
                if np.random.rand() < self.rate:
                    children[i] = self.mutation(children[i])

            # 次世代へ
            population = children.copy()

            # 最良の個体を残す
            population[0] = tmp_max_individual

            # 結果の表示
            if self.debug:
                logging.info(
                    f"Generation {generation}: Best fitness = {np.max(fitness)}, 10 metrics = {self.tovec(self.history, 10)}"
                )

            # 個体群の出力
            self.dump_population(population, generation, fitness)

        # 適応度の最小値，ターゲット，最適解，10個の指標を返す
        return (
            -1 * np.max(fitness),
            self.target,
            population[np.argmax(fitness)],
            self.tovec(self.history, 10),
        )


def main():
    """main関数．実行時にターゲットデータ（./results/synthetic_fitting_target.csv）を読み込み，GAを実行する．このファイルは`python pca.py`で生成されるので，予め実行しておく必要がある．

    Raises:
        FileNotFoundError: ターゲットデータが存在しない場合に発生
    """
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

    # ターゲットデータの読み込み．ターゲットデータ名のバリデーションはシェルスクリプト側で行われている
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
            target_data=target_data,
            population_size=population_size,
            rate=rate,
            cross_rate=cross_rate,
            history=[],
            jl_main=jl_main,
            thread_num=thread_num,
        )
        res = ga.run()
        result.append(res)
    # 適応度の最小値でソート（昇順）
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
