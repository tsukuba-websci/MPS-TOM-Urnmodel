## 概要
既存モデルである全探索を行います。
rho,nu=1,2,...,20と戦略SSW,WSWの2種類の全てのパラメータ組(20x20x2=800)で全探索します。
それぞれUbaldiのモデルを走らせ、10個の指標を計測して保存します。

GA,QDはターゲットデータに合うように(rho,nu,s)を探索しますが、`full-search/main.jl`ではターゲットデータを意識していません。
壺モデルを走らせる処理と10個の指標の計測が終わっていれば、ターゲットデータとのフィッティング（ターゲットデータとのdistanceを求め、最低な(rho,nu,s)を見つける操作）は一瞬で完了するため、`/visualize`で可視化するときに行われています。

## 実行方法
```bash
$ julia --proj=. --threads=auto main.jl
```
この結果は`./results/existing_full_search.csv`に保存されます。

内容は次の形です。同じパラメータで10回試行するため、同じ (rho, nu, s) の組の結果が10件含まれます。
実際に使用する際にはこれらの組で`groupby`した上で算術平均で集約するような操作が期待されます。
```
rho,nu,s,gamma,no,nc,oo,oc,c,y,g,r,h
<int>,<int>,<string>,<float>,<float>,<float>,<float>,<float>,<float>,<float>,<float>,<float>,<float>
...
```