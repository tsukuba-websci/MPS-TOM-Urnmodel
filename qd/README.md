# Quality Diversity(QD)によるパラメタ探索

## 概要

Quality Diversity（QD）による壺モデルのパラメータ探索を行うソースコード一式が含まれています。

この探索には、[pyribs](https://pyribs.org) というQDのライブラリを使用しています。探索空間はCVT-MAP-Elitesと呼ばれる手法によって固定数のセルに分割されています。探索空間の軸であるbehavioral descriptor(BD)には、壺モデルで生成されたネットワークをgraph2vecで変換して使用しており、値域は[-5,5]です。


## 実行方法

### behavioral descriptorのための前処理
`make_model.py` を実行してください。
```bash
$ python make_model.py
```
BDに用いられる64,128,256次元のgraph2vecのモデルが作成されます。

### QDによる探索
このディレクトリ（`/qd`）に移動した上で，`main.py` を実行してください。
次元数は`make_model.py`で設定した64,128,256次元のいずれかを指定してください。

```bash
$ pwd # => /path/to/qd
$ python main.py {twitter,aps,synthetic} <dim> <cells> [rho] [nu] [{SSW,WSW}]
```

実行例を以下に示します。
ターゲットデータが実データ(twitter)でBDが128次元、分割するセル数が500の場合：
```bash
$ python main.py twiiter 128 500
```
ターゲットデータが合成データ(rho,nu,s)=(5,5,SSW)でBDが64次元、分割するセル数が750の場合の場合：
```bash
$ python main.py synthetic 64 750 5 5 SSW
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
$ python tuning.py <target_data>  [rho] [nu] [s]
```
結果は`./results/hyperparams-search/<target>/cells<cells>/dim<dim>`以下に`main.py`の結果と同じ形式で保存されます。`tuning.py`の実行には時間がかかるため、注意してください。

チューニングした結果から最適なパラメータを見つけ、可視化できるようにするには、`copy_best.py`を実行してください。
```bash
$ python copy_best.py <target_data>  [rho] [nu] [s]
```
`results/hyperparams/`にある各ハイパーパラメータでの結果から、最良のものが`results/`にコピーされます。
