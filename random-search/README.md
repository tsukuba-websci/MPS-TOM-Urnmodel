## 概要
パラメータのランダムサーチを行います。
決められた`rho,nu,recentness,frequency`の範囲でランダムに遺伝子を生成する。
生成した遺伝子で壺モデルを一定回数実行し、遺伝子パラメータと10個の指標を保存する。

GA,QDはターゲットデータに合うように`(rho,nu,s)`を探索しますが、`randome-search/main.py`ではターゲットデータを意識していません。
ターゲットデータとのフィッティング（ターゲットデータとのdistanceを求め、最良の`(rho,nu,s)`を見つける操作）は`fitting.py`で行われます。

## 実行方法
10個の指標を計算して保存するには、以下のコマンドを実行してください。
```bash
$ python main.py
```
この結果は`./results/random-search.csv`に以下の形式で保存されます。
```
rho,nu,recentness,frequency,gamma,no,nc,oo,oc,c,y,g,r,h
<int>,<int>,<float>,<float>,<float>,<float>,<float>,<float>,<float>,<float>,<float>,<float>,<float>,<float>
...
```


次に以下のコマンドを実行してください。`./results/random-search.csv`を用いて、各ターゲットデータとの距離を計算できます。
```bash
$ python fitting.py <target_type>
```
全ての出力結果は`./results/<target>/archive.csv`に以下のような形式で出力されます。
また、最もdistanceが小さいパラメータの情報は`best.csv`に出力されます。
```
rho,nu,recentness,frequency,distance
<float>,<float>,<float>,<float>,<float>
```

