import os
from typing import Any, Tuple


class JuliaInitializer:
    """Juliaの初期化を行うクラス．"""

    def __init__(self) -> None:
        self.__setup_threads()
        self.__setup_julia()

    def initialize(self) -> Tuple[Any, int]:
        """Juliaの初期化を行い，Mainモジュールとスレッド数を返す

        Returns:
            Tuple[Any, int]: Mainモジュールとスレッド数
        """
        return self.jl_main, self.thread_num

    def __setup_threads(self):
        JULIA_NUM_THREADS = "JULIA_NUM_THREADS"
        if JULIA_NUM_THREADS not in os.environ:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                self.thread_num = cpu_count
            else:
                self.thread_num = 4  # default thread number
            os.environ[JULIA_NUM_THREADS] = str(self.thread_num)
        else:
            self.thread_num = int(os.environ[JULIA_NUM_THREADS])

    def __setup_julia(self):
        from julia.api import Julia

        Julia(compiled_modules=False)

        from julia import Pkg  # type: ignore

        Pkg.activate(".")  # use ./Project.toml
        Pkg.instantiate()  # install dependencies

        from julia import Main  # type: ignore

        self.jl_main = Main

        self.jl_main.include("../lib/history2vec.jl")
