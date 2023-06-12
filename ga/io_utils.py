import argparse
import csv
import json
import os


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """コマンドライン引数のパース．

    Args:
        parser (argparse.ArgumentParser): コマンドライン引数のパーサー

    Returns:
        argparse.Namespace: コマンドライン引数のパース結果
    """

    parser.add_argument("population_size", type=int, help="個体数")
    parser.add_argument("mutation_rate", type=float, help="突然変異率")
    parser.add_argument("cross_rate", type=float, help="交叉率")
    parser.add_argument(
        "target_data",
        type=str,
        choices=["twitter", "aps", "synthetic"],
        help="ターゲットデータ",
    )
    parser.add_argument("rho", type=int, nargs="?", default=None, help="rho")
    parser.add_argument("nu", type=int, nargs="?", default=None, help="nu")
    parser.add_argument("s", type=str, nargs="?", default=None, choices=["SSW", "WSW"], help="strategy")

    # parser.add_argument("-p", "--prod", action="store_true", default=False, help="本番実行用フラグ．出力先を変更する．")
    parser.add_argument("-f", "--force", action="store_true", default=False, help="既存のファイルを上書きする．")
    args = parser.parse_args()
    return args


def export_individual(distance: float, individual: list, fpath: str) -> None:
    """個体をCSVファイルに出力する．

    Args:
        distance (float): ターゲットとの距離
        individual (list): 個体を表すタプル．(rho, nu, recentness, frequency)の順
        fpath (str): 出力先のパス
    """
    header = ["rho", "nu", "recentness", "frequency", "objective"]
    objective = distance
    row = [*individual, objective]
    with open(fpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)


def dump_json(result: tuple, fpath: str) -> None:
    """GAの結果をJSONファイルに出力する．

    Args:
        result (tuple): GAの結果
        fpath (str): 出力先のパス
    """
    print(result)
    res = {
        "min_distance": result[0],
        "params": {
            "rho": result[2][0],
            "nu": result[2][1],
            "recentness": result[2][2],
            "frequency": result[2][3],
        },
        "target": {},
        "result": {},
    }
    for field in result[1]._fields:
        res["target"][field] = getattr(result[1], field)
        res["result"][field] = getattr(result[3], field)

    json.dump(res, open(fpath, "w"), indent=4)


def pass_run(force, fpath) -> bool:
    """GAの実行をスキップするかどうかを判定する．forceがFalseかつすでに結果が存在する場合にTrueを返す．

    Args:
        force (bool): 既存のファイルを上書きするかどうか
        fpath (str): 出力先のパス
    """
    if force:
        return False
    return os.path.exists(fpath)


def validate(population_size, rate, cross_rate) -> None:
    """GAのパラメータのバリデーション．

    Args:
        population_size (int): 個体数
        rate (float): 突然変異率
        cross_rate (float): 交叉率

    Raises:
        ValueError: population_size, num_generations, rateのいずれかが不正な値の場合に発生
    """
    if not population_size > 0:
        raise ValueError("population_size must be positive.")
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1.")
    if not 0 <= cross_rate <= 1:
        raise ValueError("cross_rate must be between 0 and 1.")
