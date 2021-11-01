第1回: チュートリアルの実行
==

「2021年度 B3研究プロジェクト 〜 PyTorchによるニューラルネットワーク実装と環境構築の演習」


## 全体の流れ

- 第1回 チュートリアルの実行 (今回)
- 第2回 モデルの可視化
- 第3回 Optunaによる自動パラメータ探索
- 第4回 演算ライブラリのカスタマイズ (Python実装)



## 概略

第1回は[Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja)を使用し、[PyTorch公式のチュートリアル](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)を進めて、ニューラルネットワークの実験の進め方を身に付けてもらうことを目的としている。


## 準備

まずはブラウザを起動し、[Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja)にアクセスする。
メニューから　*ファイル → ノートブックを新規作成* を選択する。
（このきGoogleアカウントにログインしていない場合、ログインが要求される。ログインした後、再度選択する。）

続いて、別のタブで[Google Drive](https://drive.google.com/drive/my-drive)にアクセスして、Colab Notebooksという名のディレクトリが生成されていることを確認する。
このディレクトリ直下にノートブックが保存されていくことになる。
のちに、自作のモジュールを読み込めるように`My Drive/Colab Notebooks/b3_proj_2021/MyModules`の構造でディレクトリを作成しておく。
![image.png](./fig/01_drive.png)


<!-- ## JupyterHub へのログイン

["zeus" サーバ (192.168.3.36) の 8000 番ポートで 動作している JupyterHub](http://192.168.4.36:8000/) にアクセスし、ArtIC アカウントでログインする。

Jupyter はブラウザ上で Python のプログラムを書き、逐一実行して試行ができる実験環境である。
トライ＆エラーで１カ所ずつ試し、必要に応じて内部状態を可視化するという過程を繰り返すニューラルネットワーク構築・学習の教育及び初期デバッグに最適である。
一方で「プログラムに書いた処理を延々と回し続ける」という本来のプログラミングの目的にはやや不向きなので、数時間〜数日単位で時間のかかる本格的なネットワーク構築のフェーズに移ったら Jupyter ではなく通常の Python を使って実験を行う（こちらに関しても演習を用意する）。

ここに、画面左側のファイルタブから先ほど作成したシンボリックリンクを辿り、 `~/work/Jupyter/Exercise01_01` 以下に作業ディレクトリを作成する。
**図中のパスはスペルミスしているので注意。 Jupyter が正しい。** ~~やっぱり眠気を押払ってマニュアルなんて書くもんじゃない。~~
![image.png](/attachment/5e8b2ae9433e41004bf06c8b)

次に作成したディレクトリに入り、 notebook を作成する。
![image.png](/attachment/5e8b2c1a433e41004bf06c8c) -->
ここで、Google　Colabのタブに戻る。
作成されるファイルは `Untitled` という名前になるので、これを rename して `01_tutorial` 等、試した順番と内容がわかる名前に変更する。


![image.png](./fig/01_rename.png)

以降、現れたセルにコードを打ち込み、 Shift-Return で逐次実行、実行ボタンで一括実行ができる。
これでエラーやバグと戦いながらチュートリアルを進めてゆく。
なお最初から実行したくなった場合は「ランタイムを再起動」を押してカーネルをリセットする。これをしないとローカル変数等のデータが残ったままとなり、またモジュールの import は２度実行されないので、結果が正しくならない場合がある。
<!-- ![image.png](/attachment/5e8b2e79433e41004bf06c8e) -->

## 公式チュートリアルの進行

[https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)

これに従い、説明を読んでコードのそれぞれが何をやっているのか理解しながら進めてゆく。
セルごとに実行されるので失敗上等、間違えたら修正してそのセルだけ Shift-Return で再実行、迷ったら「ランタイムを再起動」で最初から。

なおデータセットは初回実行時にダウンロードされる。チュートリアル中に出現する
```python:
datasets.CIFAR10(root='./data', train=......)
```
は`./data`にデータセットを保存することを意味するが
このディレクトリはランタイムが切断されるたびに初期状態にリセットされデータが消える。
実行時に毎回ダウンロードする(数秒かかる)のを防ぐために、自身のGoogle Driveにデータセットを保存することができる。
![image.png](./fig/01_dataset.png)
図の手順でGoogle Driveに接続するとランタイムの接続が切れてもデータが保存されたままになる。

```python:
datasets.CIFAR10(root='./drive/My Drive/Colab Notebooks/b3_proj_2021/data', train=......)
```
のように書き替えることでGoogle Drive上に保存できる。
ただし、データセットの容量は300MB程度なので、空き容量(無料だと全容量15GB)と相談して選んで欲しい。（今後の演習では自分のDriveから読み込む設定になっている。
元のままでも動くが最初にデータセットのダウンロードが入るため少し時間がかる。）

また、途中の
```python:
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```
は毎度
```python:
PATH = './cifar_net_01.pth'
torch.save(net.state_dict(), PATH)
```
等に変えて実行すると上書きされない。
先ほどと同様にGoodle Drive上のディレクトリを指定することで手元に保存しておくこともできる。


### GPU での実験

一通り認識精度まで出て
> Training on GPU
>
にすすむ際には、一度通常の保存をしたあと *ファイル → ドライブにコピーを保存* でファイル名を `02_tutorial_gpu.ipynb` としてファイルを分けて保存し、*編集 → 出力を全て消去* として出力を消す。

今 Colab が動作している計算サーバはデフォルトではCPU実行になっている。
GPUを利用するにはノートブックごとに設定する必要がある。
![image.png](./fig/01_gpu.png)
 *ランタイム → ラインタイムのタイプを変更 → GPU*　を選択することでGPUを使用することができる。
これはノートブック毎に設定しなければならないので、次回以降忘れないように注意する。

最初の `import` のセルを実行したあと、左上の *+コード* ボタンを押して空のセルを作る。
![image.png](./fig/01_addcell.png)

作ったセルにチュートリアルの
```python:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
```
を書く。 出力が"cpu"ではなく"cuda:0"となっていればGPUが使用できている。"cuda:0"の0はGPU番号を表している。複数枚GPUがある環境で数字を変えることで対応するGPUで計算することができる。(Google ColabではGPUを1枚しか利用できない。)

次に、チュートリアルに従い
```python:
net = Net()
```
のあと（同一セルでも、新しくセルを作っても良い）に
```python:
net.to(device)
```
を追加する。

同様に学習ループの
```python:
for epoch in range(2):  # loop over the dataset multiple times
    .......

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
```
を
```python:
for epoch in range(2):  # loop over the dataset multiple times
    .......

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
```
に変えて実行する。

どうだろう、速くなっただろうか？
実感はあまりないと思う。
それはバッチサイズもネットワークの規模が小さいから並列化の効果が見えにくいせいである。


## モデルのカスタマイズ

ここまでくればモデルを組み換えてより精度の高いモデルを作ることができるはずである。
別ファイルに保存して続けてみよう。


### 学習可能な `Module` のメカニズム

`Net` クラスの定義を見てみると
```python:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
となっている。

学習可能な（i.e. 誤差逆伝播法の自動微分が定義された）層は `torch.nn.Module` クラスのサブクラスとして実装し、
内部で使う演算のクラス（この例でいう `nn.Conv2d`, `nn.MaxPool2d`, `nn.Linear` …… これらも `torch.nn.Module` クラスのサブクラスである）のインスタンス化や係数の alloc （この場合は `Conv2d` と `Linear` のインスタンスが内部的に重み係数を持っている）はイニシャライザである `__init__()` メソッドで行う。

定義された演算を使って実際に計算を進めるのは `forward()` メソッドで定義する。
これは学習ループの内部の
```python:
outputs = net(inputs)
```
で呼ばれる。
またここで使う演算 [^1] は内部的に `backward()` メソッドを持っており、後ほど
```python:
loss = criterion(outputs, labels)
loss.backward()
```
の内部で呼ばれている。
これで計算された微分値が係数の更新に使用されている。

`Module` は `forward()` の入口と出口の情報を持っており [^2] 、その中で使った演算の微分と係数の更新は自動的に行われるので明示的に書く必要はない。
後々自分で演算（Quantizer など）を定義する際は自分で `backward()` を実装しなければならなくなる。


[^1]: 正確には `Conv2d` 等の `nn.Module` サブクラスのインスタンスではなく、さらにその内部で使用されている `autograd.Function` サブクラス

[^2]: 正確には `Optimizer` の内部と `autograd.Function.forward()` メソッドが協調的に入力の途中計算結果を保存し、 `autograd.Function.backward()` メソッドが `Tensor` が持っている出口の微分値を使って入口の微分値を計算する


### 自分のモデルを作る
`Conv2d` のイニシャライザ引数は `Conv2d(入力チャネル, 出力チャネル, カーネルサイズ)` である ([公式リファレンス](https://pytorch.org/docs/stable/nn.html#conv2d))。
チュートリアルでは使っていないが、 `floor(カーネルサイズ / 2)` （つまりカーネルサイズが3だったら1、5だったら2）を `padding` 引数に指定し、畳込みの前後で shape が変わらないようにするのが一般的である（この設定を資料によって、具体的には TensorFlow などは *valid convolution* と呼んでいる）。
必要ならば `stride` 引数を指定してストライドを使うこともできる。
```python:
self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 入力3→出力32, カーネル3, パディング1
```

`Linear` は `Linear(入力次元, 出力次元)` である ([公式リファレンス](https://pytorch.org/docs/stable/nn.html#linear))。

`MaxPool2d` は `MaxPool2d(カーネルサイズ, ストライド)` だが、これは近代的なネットワークであれば `MaxPool2d(2, 2)` で固定と思って良い。

`forward()` の途中にある
```python:
x = x.view(-1, 16 * 5 * 5)
```
は、 `(バッチサイズ, チャネル, 高さ, 幅)` の4次元配列となっている `Conv2d` 出力（データセットから読み込んだ画像もこの形式）を `(バッチサイズ, 次元)` の2次元配列に整形する意味がある。 `-1` は shape 指定に１つまで含めることができ、その次元は全体のサイズと残りの次元サイズから計算せよという意味で、バッチサイズは実行してみるまでわからないのでこういう書き方をする慣習である。

これを念頭に、モデルを大きくして実験してみよう。

### バッチサイズとエポック数

```python:
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

......

testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=2)

```
程度に、バッチサイズを大きくすると学習が速い。
この際は学習ループの中も
```python:
if i % 100 == 99:    # print every 100 mini-batches
    print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 100))
    running_loss = 0.0
```
に変更する。

また学習ループの
```python:
for epoch in range(2):  # loop over the dataset multiple times
```
の数値を変えると複数回繰り返してデータセットを流しながら学習がすすむ（エポック数という）。
一般にはモデルの傾向を評価するためには 10 〜 20 、実用目的には 100 以上回す。当然時間はかかる。
**今回はまだ学習の途中経過を表示できる機構を入れていないのであまり大きくすることは推奨しない。**

### Batch normalization の利用
どうだろう、大きくしたら精度は上がっただろうか？案外難しかったのではないかと思う。
次に考えるのは Batch normalization である。
これは特に深いネットワークの発散を抑える効果がある。

PyTorch においては Convに `nn.BatchNorm2d(チャネル数)` 及び Linear には `nn.BatchNorm1d(次元)` が使える ([公式リファレンス](https://pytorch.org/docs/stable/nn.html#normalization-layers)) 。
以下を参考に、 `__init__()` と `forward()` に組み入れる。
この例における `MaxPool2d` と異なり、 Batch Normalization は内部状態を持つので、各層ごとに別のインスタンスを作る必要がある。

```python:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_conv3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_conv4 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn_conv1(self.conv1(x)))
        x = self.pool(F.relu(self.bn_conv2(self.conv2(x))))
        x = F.relu(self.bn_conv3(self.conv3(x)))
        x = self.pool(F.relu(self.bn_conv4(self.conv4(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x
```

## 課題

今回のチュートリアルはここまでである。
わからない部分は自学を求める。

<!-- 他は、 Visual Code Studio でサーバに接続して Python をコーディングできるまでの準備、及び Verilog の実験環境がまだできていない場合はそれも進めていただきたい。 -->

### VGG11 の実装とカスタマイズ
VGG11とはモデル構造の名称である。シンプルな構造であるがゆえに広く使われる。
ArtICのGitHub [https://github.com/ArtIC-TITECH/b3-proj-2021](https://github.com/ArtIC-TITECH/b3-proj-2021) の `Exercise01/01_01_vgg.ipynb` に少しUIを改善したチュートリアルを置いた。Google Colabにて *ファイル → ノートブックを開く → GitHub* と進み上記のURLを入力すると開ける。
これを自分のDriveに保存して、改造しVGG11モデルを作って学習させてみると良い。
実行時にランタイムの種類をGPUにするのを忘れずに。
既に上で述べたモデル改造の要素は入れてあるが、わざと未完成にしてあるので、 [参考ページ](https://qiita.com/MuAuan/items/86a56637a1ebf455e180) 等をみながらVGG11（参考サイトのTable 1の左端の列。 *Conv3-512* は *カーネルサイズ3, 出力チャネル512のConv2d* を意味している）を実装してみる。

それが終わったら（おそらくそのままではあまり思った精度は出ないので）別ファイルに保存し、モデルをカスタマイズをしてみる。


