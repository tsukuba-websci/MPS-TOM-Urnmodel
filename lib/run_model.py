from dataclasses import dataclass
from typing import List, Tuple

from rsurn import Environment, Gene


@dataclass
class Params:
    rho: float
    nu: float
    recentness: float
    frequency: float
    steps: int


def run_model(params: Params) -> List[Tuple[int, int]]:
    """ベクトル化壺モデルを実行する．

    Args:
        params (Params): パラメータ

    Returns:
        List[Tuple[int, int]]: 相互やりとりの履歴
    """
    rho = int(params.rho)
    nu = int(params.nu)
    gene = Gene(rho, nu, params.recentness, params.frequency)
    env = Environment(gene)
    for _ in range(params.steps):
        caller = env.get_caller()
        callee = env.get_callee(caller)
        env.interact(caller, callee)
    return env.history
