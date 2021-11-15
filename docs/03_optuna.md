第3回: Optunaによる自動パラメータ探索
===

「2021年度 B3研究プロジェクト 〜 PyTorchによるニューラルネットワーク実装と環境構築の演習」


## 概略

前回まで、モデルの探索は手動で行っていた。
今回はこれを自動化するツールであるOptunaを利用する。


## 関連資料
- [Optuna公式](https://optuna.org/)
- [Optuna公式ドキュメント](https://optuna.readthedocs.io/en/stable/)



## 準備

### プログラムのデプロイ
Google Colabで以下のファイルを開き自分のGoogle Driveにコピーを保存する。
[ArtIC GitHub](https://github.com/ArtIC-TITECH/b3-proj-2021)の
[`Exercise03/03_01_template.ipynb`](https://github.com/ArtIC-TITECH/b3-proj-2021/blob/master/Exercise03/03_01_template.ipynb)
[`Exercise03/03_02_template_prune.ipynb`](https://github.com/ArtIC-TITECH/b3-proj-2021/blob/master/Exercise03/03_02_template_prune.ipynb)

また、[util.py](https://github.com/ArtIC-TITECH/b3-proj-2021/blob/master/Exercise03/util.py)を`My Drive/Colab Notebooks/b3_proj_2021/MyModules'`にアップロードする。
方法例として
1. Google ColabでGoogle Driveをマウント
2. `!touch ./drive/My\ Drive/Colab\ Notebooks/b3_proj_2021/MyModules/util.py`をセルで実行
3. 左のメニューバーから`util.py`を探してダブルクリックすることで、右側にエディタを開く。
3. 別のタブで[util.py](https://github.com/ArtIC-TITECH/b3-proj-2021/blob/master/Exercise03/util.py)を開き、本文全体をコピーして、Colabのエディタ(右側)に貼り付けて保存する。(Mac: ⌘+s, Windows: Ctrl+s)



### SQLサーバの設定
Optunaが途中で試したパラメータの経過はデータベースに記録される。
これこそOptunaが強力な理由の一つ、同じデータベースサーバを複数の学習プロセスが参照すると、それだけで並列探索となる。
本演習ではSQLiteというファイルベースのデータベースエンジンを用いるためSQLサーバは設定せずに利用可能である。

<!-- Optunaが途中で試したパラメータの経過はデータベースに記録される。
これこそOptunaが強力な理由の一つ、同じデータベースサーバを複数の学習プロセスが参照すると、それだけで並列探索となる。

それに伴い、まず設営済みのMySQLサーバのユーザを設定する。

1. 適当なUbuntu計算サーバ（pollux, selene, artemis, zeus）にSSHログインする。
2. 次のコマンドにより、MySQLサーバにログインする。
    ユーザ名、初期パスワード、データベース名全て **ArtICユーザ名の `-` を `_` に置き換えたもの** になっている。
    例: `j-smith` さんの場合
    ```
    Server:~$ mysql -P 53306 -h zeus -u j_smith -p
    Enter password:     ← j_smith が初期パスワード
    ```
3. SQLサーバに以下のコマンドを打ち込み、パスワードを変更する。
    結果は必ず表示 `Query OK` を確認すること。
    ```
    mysql> SET PASSWORD='新しいパスワード';
    Query OK, 0 rows affected (0.00 sec)
    ```
4. Ctrl-Dを入力して抜ける。 -->


## 演習課題の進行

### テンプレ解説
テンプレートは前回まで使用したものを整理したものとなっている。主な差分は以下。

- 画像の表示、精度計算等の関数は `util.py` に分離
- VGGモデル定義を関数と `nn.Sequential` を使って整理
- Optunaでのループ構造に対応するため、学習ルーチンを `train()` 関数にまとめる
- 中間層のActivationをTensorBoardに表示するためのトリックを追加（ `train()` 関数の `util.IntermediateOutputWriter` の部分）
- `transform`, `lr_scheduler` をより実際的なものに変更

### Optunaのインストール
Google ColabにはOptunaがインストールされていないため、最初のセルで以下のコマンドを実行する。
```
!pip install optuna
```
ランタイムの接続がきれる度にインストールする必要がある。
演習ではOptunaを使うのは今回だけだが、毎回わずらわしければGoogle Driveに保存してそこから読み込むこともできる。
[外部サイト](https://ggcs.io/2021/06/22/google-colab-pip-install/)

### Optuna対応の基本
これを順に進めてもらう。
詳細は演習セッションで説明するが、Optuna対応の基本は以下のような手順となる。

- １つのパラメータ集合を受け取って学習し、評価するまでを一つの関数にまとめる
- その関数は `objective(trial)` というように `optuna.Trial` オブジェクトを引数に受け、学習と評価を行った結果を返す（目的関数と呼ばれている）
    - objective関数内で `trial.suggest_int("param_name", min, max)` のようにsuggest系関数を呼ぶと、これまでの試行から考慮されたパラメータが "降ってくる"
    - あとはこれをプログラム内でベタ書きした定数と同じように扱って学習を回す
- 「ある試行で選択されるハイパーパラメータの集合」が `Trial` で、複数の試行＝複数のTrialの全体集合は `optuna.Study` オブジェクトが管理している
- こちらが行うのは、objective関数とStudyを作って `study.optimize(objective, n_trials)` 関数を呼ぶだけ
- Studyが `n_trials` 回だけTrialを作って `objective(trial)` を呼ぶ。今回のプログラムでは `objective()` が `Net` を降ってきたパラメータで生成し、 `train()` 関数を呼ぶようになっている


というわけで、まず最初にやってもらうのはセル6の `train` 関数で定義している学習率とOptimizerアルゴリズムをOptunaに任せて選択するように調整することである。
```python:
if optim_type == 'adam':
    lr = 1e-3  # trial.suggest_........("lr_adam")
```
となっているのを、trialから提案してもらうように書き換えてみよう。
それが済んだら一度実行してみる。

次はOptimizerもAdamとSGDで選択できるようにしたり、モデル構造自体を選択するようにもしてみると良い。
使用できるsuggest関数は [公式リファレンス](https://optuna.readthedocs.io/en/stable/reference/trial.html) をみるのが早いだろう。

**注意: `Study` オブジェクトは `study_name` 引数でデータベース上のStudyを一意に区別している。したがって別の意図を持って別のパラメータチューニングを回す際はこれを変える必要がある。** さもなくば無関係であるはずの複数Trialが同じ探索空間に乗ってしまう。本テンプレートにおいてはセル3の `study_name` がそれに対応しているので、適宜変更して進めること。

### Prunerの使用

テンプレートは
[ArtIC GitHub](https://github.com/ArtIC-TITECH/b3-proj-2021)の
[`Exercise03/03_02_template_prune.ipynb`](https://github.com/ArtIC-TITECH/b3-proj-2021/blob/master/Exercise03/03_02_template_prune.ipynb)を使用する。

Optunaのもう一つ強力な機能にPrunerという機能がある。学習の進行中に、過去の学習Trialの履歴と比較して精度が良くないと判断されるとそのTrialを中断して次の試行に移るという機能である。膨大なパラメータ空間を扱う際に有用である。

この使い方の基本は以下の通りである。 [公式チュートリアル](https://optuna.readthedocs.io/en/stable/tutorial/pruning.html) も参照。

- `optuna.Pruner` クラスのサブクラスのインスタンスを生成。各種アルゴリズムが揃っている（[公式リファレンス](https://optuna.readthedocs.io/en/stable/reference/pruners.html)）
    - `MedianPruner`: 同一エポックの結果における中央値より低いTrialを中断
    - `PercentilePruner`: 同一エポックの結果における上位%以下のTrialを中断
    - `SuccessiveHalvingPruner`: Successive Halving アルゴリズムによるリソース最適化
    - `ThresholdPruner`: 結果が上下界をはみ出したら中断
- それを `create_study()` 関数に渡す
- 学習ルーチンのTest精度評価にて、 `trial.report(train_acc, epoch)` として中間結果を報告
- 学習ルーチンのEpochループの最後で `trial.should_prune()` を呼ぶ
    - この結果が `True` だったらそのTrialは中断すべきという判断。 `raise optuna.exceptions.TrialPruned()` として例外を投げる
    - するとそのTrialは終了され、次のTrialに移る
    - 例外なので、例えばモデルの保存のような後処理が必要ならtry-except-finallyで囲む

## 宿題

自分で前回作ったモデルを今回のように変数で層数を変えたりできるように改変してみよう。
余力があるなら別のネットワークモデルを調べて実装してみるとか、もっと違うパラメータを振ってみるとか。
