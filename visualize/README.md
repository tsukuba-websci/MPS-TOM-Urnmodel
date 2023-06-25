# full searh, GA, QDの実験結果の可視化

## 概要

以下の3種類の図を作成する。
- 棒グラフ
- レーダーチャート
- タイムライン

## 実行方法

### 前処理
GA,QDで見つけた最良のrho,nu,recentness,frequencyの組で壺モデルを10回走らせます。
```bash
$ pwd # => /path/to/visualize
$ python preprocessing.py <target_type>
```
この結果は`./results/fitted`に保存され、棒グラフ・レーダーチャートの生成に用いられます。

### 可視化のスクリプト
```bash
$ pwd # => /path/to/visualize
$ python main.py <graph_type> <target_type>
```

各引数の詳細などは `python main.py -h` あるいは `python main.py --help` で確認できます。

実行を行うと，`./results` 以下にグラフの種類ごとに分けられて結果が保存されます。