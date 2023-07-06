import os
import pickle

import networkx as nx
from karateclub.graph_embedding import Graph2Vec
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from lib.julia_initializer import JuliaInitializer
from lib.run_model import Params, run_model

if __name__ == "__main__":
    jl_main, thread_num = JuliaInitializer().initialize()
    start = 1
    stop = 21
    step = 1

    graphs = []
    tqdm_bar = tqdm(total=((stop - start) // step) ** 2, desc="Simulation")
    for rho in range(start, stop, step):
        for nu in range(start, stop, step):
            graph = nx.Graph()
            params = Params(rho, nu, 0.5, 0.5, 20000)
            for edge in run_model(params):
                graph.add_edge(edge[0], edge[1])
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_graph = graph.subgraph(largest_cc)
            largest_graph = nx.convert_node_labels_to_integers(largest_graph)
            graphs.append(largest_graph)
            tqdm_bar.update(1)
    tqdm_bar.close()

    dims = [64, 128, 256]
    for dim in dims:
        dir = f"models/dim{dim}"
        os.makedirs(dir, exist_ok=True)

        graph2vec_model = Graph2Vec(dimensions=dim)
        graph2vec_model.fit(graphs)
        graph_features = graph2vec_model.get_embedding()

        standardize_model = StandardScaler()
        std_graph_features = standardize_model.fit_transform(graph_features)

        with open(f"{dir}/graph2vec.pkl", "wb") as f:
            pickle.dump(graph2vec_model, f)
        with open(f"{dir}/standardize.pkl", "wb") as f:
            pickle.dump(standardize_model, f)
