import os
import pickle
from pathlib import Path
from typing import List, Tuple, Union, cast

import networkx as nx
import numpy.typing as npt
from karateclub.graph_embedding import Graph2Vec
from sklearn.preprocessing import StandardScaler

History = List[Tuple[int, int]]


class History2BD:
    graph2vec_model: Graph2Vec
    standardize_model: StandardScaler

    def __init__(
        self,
        graph2vec_model_path: Union[str, os.PathLike],
        standardize_model_path: Union[str, os.PathLike],
    ) -> None:
        if isinstance(graph2vec_model_path, str):
            graph2vec_model_path = Path(graph2vec_model_path)

        if isinstance(standardize_model_path, str):
            standardize_model_path = Path(standardize_model_path)

        with open(graph2vec_model_path, "rb") as f:
            self.graph2vec_model = pickle.load(f)

        with open(standardize_model_path, "rb") as f:
            self.standardize_model = pickle.load(f)

    def run(
        self,
        history: Union[History, List[History]],
    ) -> npt.ArrayLike:
        """相互作用履歴を主成分に変換する

        Parameters
        ----------
        history : History | List[History]
            相互作用の履歴データ、またはその配列

        Returns
        -------
        array-like
            変換された主成分
        """

        G: List[nx.Graph] = []
        if isinstance(history[0], list):
            G += list(map(lambda h: cast(nx.Graph, nx.from_edgelist(h)), history))
        else:
            G.append(cast(nx.Graph, nx.from_edgelist(history)))

        G = list(map(self.__make_nodes_consecutive, G))

        g_vec = self.graph2vec_model.infer(G)
        std_g_vec = self.standardize_model.transform(g_vec)
        return std_g_vec

    @classmethod
    def __make_nodes_consecutive(cls, g: nx.Graph) -> nx.Graph:
        return nx.relabel_nodes(g, dict(zip(g.nodes, range(len(g.nodes)))))
