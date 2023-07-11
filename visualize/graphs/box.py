import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

targets = ["twitter", "aps", "mixi"]
algorithms = ["full-search", "ga", "qd", "random-search"]
generation = 500

data_list = []

for target in targets:
    for algorithm in algorithms:
        if algorithm == "ga" or algorithm == "qd":
            filenum_str = "{:0>8}".format(generation - 1)
            file_path = f"../{algorithm}/results/{target}/archives/{filenum_str}.csv"
        elif algorithm == "full-search" or algorithm == "random-search":
            file_path = f"../{algorithm}/results/{target}/archive.csv"
        try:
            data = pd.read_csv(file_path)
            data["algorithm"] = algorithm
            data["target"] = target
            data_list.append(data)
        except FileNotFoundError:
            print(f"File not found: {file_path}")

combined_data = pd.concat(data_list)

sns.set(style="ticks")
sns.boxplot(x="target", y="distance", hue="algorithm", data=combined_data)
os.makedirs("results/box", exist_ok=True)
plt.savefig("results/box/empirical.png")
plt.show()
