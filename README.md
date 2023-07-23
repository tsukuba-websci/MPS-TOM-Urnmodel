# MPS-TOM-Urnmodel

Iwahashi, Okabe and Suda's experiments for MPS-TOM 2023

## Abstract
情報処理学会論文誌:数理モデル化と応用に投稿した論文に関するリポジトリです。

エージェントベースの壺モデルの探索手法(全探索,ランダムサーチ,GA,QD)の比較を行います。
- [**`data/`**](/data/)
  - 探索に用いるターゲットデータ
- [**`full-search/`**](/full-search/)
  - 既存モデルの全探索を行うスクリプト
- [**`random-search/`**](/random-search/)
  - ランダムサーチによる探索を行うスクリプト
- [**`ga/`**](/ga/)
  - 遺伝的アルゴリズムによる探索を行うスクリプト
- [**`qd/`**](/qd/)
  - Quality Diversityによる探索を行うスクリプト
- [**`visualize/`**](/visualize/)
  - 探索結果をもとに可視化するスクリプト

## Requirements
- Julia (1.8.x)
  - PyCallパッケージ：JuliaにPyCallをインストールしてください
- Python (3.9.x)
  - 必要な依存関係はpyproject.tomlに定義してあるので、それらをインストールしてください
- Rust (latest)
  - Rustの実行環境があれば問題ありません

`full-search/`,`random-search/`,`ga/`,`qd/`から上位のディレクトリ`lib/`の中身をimportするためにpoetryを用いて設定しています。依存関係は`$ poetry install`でインストールしてください。


## Usage
### Make synthetic target data
合成データの探索に用いられるデータを作成するために`make_synthetic_target.jl`を実行してください。探索時のターゲットデータとして合成データを用いる場合は先に実行しておく必要があります。
```bash
julia --proj=. --threads=auto make_synthetic_target.jl
```
結果は`./data/synthetic_target.csv`に保存されます。合成データのターゲットとなるのは以下の6つのパラメータ組です。
$$(\rho,\nu,s)=(5,5,SSW),(5,5,WSW),(5,15,SSW),(5,15,WSW),(20,7,SSW),(20,7,WSW)$$

### Search　Params
各ディレクトリ`full-search/`,`random-search/`,`ga/`,`qd/`のREADMEの指示に従って、`main.py`及び`main.jl`を実行してください。
#### Required Files
- `./data/aps.csv`
- `./data/twitter.csv`
- `./data/synthetic_target.csv`

### Visualize
`visualize/`のREADMEの指示に従って、`main.py`を実行してください。
#### Required Files
- `./data/...`
- `./full-search/results/...`
- `./random-search/results/...`
- `./ga/results/...`
- `./qd/results/...`


## License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
