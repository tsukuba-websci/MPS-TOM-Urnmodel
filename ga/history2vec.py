from dataclasses import dataclass
from typing import Any, List, NamedTuple, Tuple


@dataclass
class Params:
    rho: float
    nu: float
    recentness: float
    frequency: float
    steps: int


class History2VecResult(NamedTuple):
    gamma: float
    no: float
    nc: float
    oo: float
    oc: float
    c: float
    y: float
    g: float
    r: float
    h: float


class History2Vec:
    """相互やり取りの履歴から10この指標を計算する．実態はJulia実装history2vec.jlを呼び出すラッパー．"""

    def __init__(self, jl_main: Any, thread_num: int = 1):
        self.jl_main = jl_main
        self.thread_num = thread_num

    def history2vec(self, history: List[Tuple[int, int]], interval_num: int) -> History2VecResult:
        # もし履歴が0-originだったら1-originに変換する
        if any(map(lambda row: row[0] == 0 or row[1] == 0, history)):
            history = list(map(lambda row: (row[0] + 1, row[1] + 1), history))
        nt = self.jl_main.history2vec(history, interval_num)
        return History2VecResult(
            c=nt.c,
            g=nt.g,
            gamma=nt.gamma,
            h=nt.h,
            nc=nt.nc,
            no=nt.no,
            oc=nt.oc,
            oo=nt.oo,
            r=nt.r,
            y=nt.y,
        )
