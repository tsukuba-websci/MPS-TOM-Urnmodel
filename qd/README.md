# Quality Diversity(QD)によるパラメタ探索

## 概要

Quality Diversity（QD）による壺モデルのパラメータ探索を行うソースコード一式が含まれています。

この探索には、[pyribs](https://pyribs.org) というQDのライブラリを使用しています。探索空間はCVT-MAP-Elitesと呼ばれる手法によって固定数のセルに分割されており、今回はセル数を500に設定しています。探索空間の軸であるbehavioral descriptorには、壺モデルで生成されたネットワークをgraph2vecで128次元のベクトルに変換して使用しています。したがって、探索空間はgraph2vecによって生成されるベクトルの値域が[-5, 5]の128次元に設定されています。


## 実行方法

### behavioral descriptorのための前処理
`make_model.py` を実行してください。
```bash
$ python make_model.py
```

### QDによる探索
このディレクトリ（`/qd`）に移動した上で，`main.py` を実行してください。

```bash
$ pwd # => /path/to/qd
$ python main.py {twitter,aps,synthetic} [rho] [nu] [{SSW,WSW}]
```

実行例を以下に示します。
ターゲットデータが実データ(twitter)の場合：
```bash
$ python main.py twiiter
```
ターゲットデータが合成データ(rho,nu,s)=(5,5,SSW)の場合：
```bash
$ python main.py synthetic 5 5 SSW
```

実行を行うと，`./results/{twitter,aps,synthetic}` 以下に結果が保存されます。その中に各世代ごとのアーカイブデータと，最終的な結果が保存されます。
アーカイブデータは各世代ごと以下のような形式で、個体数(=占有されたセル数)分の行がdistance昇順にソートされた状態で出力されます。
```
rho,nu,recentness,frequency,distance
<float>,<float>,<float>,<float>,<float>
...
```
最終的な結果`best.csv`は以下のような形式で、最もターゲットデータとのdistanceが小さい個体の情報のみが出力されます。
```
rho,nu,recentness,frequency,distance
<float>,<float>,<float>,<float>,<float>
```

### ハイパーパラメータチューニング
ターゲットデータごとに適したセル数,graph2vecで表現する次元数を探索する場合は、`tuning.py`を実行してください。
```bash
$ python tunig.py <target_data>  [rho] [nu] [s]
```
結果は`./results/hyperparams-search/<target>/cells<cells>/dim<dim>`以下に`main.py`の結果と同じ形式で保存されます。`tuning.py`の実行には時間がかかるため、注意してください。

チューニングした結果から最適なパラメータを見つけ、可視化できるようにするには、`search_best.py`を実行してください。
```bash
$ python copy_best.py <target_data>  [rho] [nu] [s]
```
`results/hyperparams/`にある各ハイパーパラメータでの結果から、最良のものが`results/`にコピーされます。
