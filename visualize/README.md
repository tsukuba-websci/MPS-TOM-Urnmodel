# 既存モデルの全探索, ランダムサーチ, GA, QDの実験結果の可視化

## 概要

以下の5種類の図と2種類の表を出力するソースコードです。

図を生成するスクリプトは`/graphs`にあります。
- 箱ひげ図
  - `box.py`
  - full-search,random-search,GA,QD
  - Fig.3(A), Fig.5(A)
- タイムライン
  - `timeline.py`
  - GA,QD
  - Fig.3(B), Fig.5(B)
- レーダーチャート
  - `radar_chart.py`
  - full-search,GA,QD
  - Fig.4, Fig.6
- ネットワークのベクトルが作るマップ
  - `map.py`
  - GA,QDの実データ
  - Fig.7
- 棒グラフ
  - `bar_graph.py`
  - full-search,random-search,GA,QD

表を生成するスクリプトは`/tables`にある。
- 最良個体の距離の平均と分散を示す表
  - `print_best.py`
  - full-search,random-search,GA,QD
  - Table1, Table2
- 最良の遺伝子5つを示すlatex形式の表
  - `latex_table.py`
  - QD
  - Table3


## 実行方法

### 前処理
GA,QD,ランダムサーチで見つけた最良の`(rho,nu,recentness,frequency)`の組で壺モデルを10回走らせます。（既存モデルの全探索では初めから10回走らせているため、ここでは行いません）
```bash
$ pwd # => /path/to/visualize
$ python preprocessing.py <target_type>
```
この結果は`./results/fitted`に保存され、棒グラフ・レーダーチャート,最良個体の距離の表の生成に用いられます。


GA,QDの最終的な全個体のに対して、壺モデルを回してネットワークを生成します。そのネットワークをgraph2vecしたベクトルを保存しておきます。
```bash
$ python make_vec.py <target_type>
```
この結果は`./results/vec`に保存され、ネットワークのベクトルが作るマップに用いられます。



### 可視化のスクリプト
図を出力する場合:
```bash
$ pwd # => /path/to/visualize
$ python main.py <graph_type> <target_type>
```
各引数の詳細などは `python main.py -h` あるいは `python main.py --help` で確認できます。
実行を行うと，`./results` 以下にグラフの種類ごとに分けられて結果が保存されます。


最良個体の距離の平均と分散を示す表を出力する場合:
```bash
$ python tables/print_best.py <target_type>
```
また、latex形式で遺伝子の表を出力するには、以下のコマンドを実行してください。
```bash
$ pwd # => /path/to/visualize
$ python graphs/latex_table.py
```
`MPS-TOM-urnmodel/`で用意されているpoetry環境では、`latex_table.py`だけ`pandas`のversionの依存関係が合わないため、動きません。
`pandas>=1.5.3`, `Jinja2>=3.0.0`, `matplotlib`, `numpy`のある環境で実行してください。
