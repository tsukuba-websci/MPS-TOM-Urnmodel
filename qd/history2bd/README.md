# History2BD
相互作用の履歴をbehavioral descriptor(QDの軸)にマッピングするモジュール
(caller, callee)のペアの配列をグラフに変換し、グラフからベクトルに変換する

## I/O
### Input
#### モデル生成時
次のファイルへのパス

- graph2vec.pkl
- standardize.pkl

#### 実行時
相互作用の履歴

(caller, callee) のペアの配列のこと。具体的な型は `List[Tuple[int, int]]` 。

単体でも、複数をリストにまとめたもの (`List[List[Tuple[int, int]]`) でもOK。

### Output
入力された相互作用の履歴を主成分に変換したベクトル。

- n1: 入力した履歴の本数（単体で入力した場合は1本）
- n2: graph2vecで生成したベクトルの次元数(128次元)

としたとき、 n1*n2 の Numpy.NDArray が返る。

## Example

```python
from history2bd import History2BD, History
import pickle

with open("history.pkl") as f:
    # typed: History == List[Tuple[int, int]]
    history = pickle.load(f)

history2bd = History2BD(
    "graph2vec.pkl",
    "standardize.pkl",
)

result = history2bd.run(history)
```