## 回归方法的房价预测

我们在这一节用一个竞赛来检验我们上一阶段的学习成果。

> [Kaggle](https://www.kaggle.com/)是一个当今流行举办机器学习比赛的平台， 每场比赛都以至少一个数据集为中心。 许多比赛有赞助方，他们为获胜的解决方案提供奖金。 该平台帮助用户通过论坛和共享代码进行互动，促进协作和竞争。 虽然排行榜的追逐往往令人失去理智： 有些研究人员短视地专注于预处理步骤，而不是考虑基础性问题。 但一个客观的平台有巨大的价值：该平台促进了竞争方法之间的直接定量比较，以及代码共享。

我们使用的是house-price-advanced-regression-techniques这个项目。



### 思路理顺

#### step1 这个项目的目的是什么

kaggle给了我们两个数据集，里边有房屋的各项特征（有部分项目是缺失的）和价格，我们需要**通过数据预处理和设计模型**，**学习训练集里的数据**，并在**测试集上预测房价**。

#### step2 我们需要的基础

这个项目对硬件的需求比较小，cpu已经足够跑了，在前一阶段的学习里我们知道了不同网络的架构，因为我们是初学者，我们可以先对自己要求低一些，**通过成品的数据预处理来熟悉过程**，能理解这一个过程里发生了什么，这需要你知道**os/pandas/numpy**的部分内容(下文中会进行提到，但是如果你有基础会阅读的更为轻松)。

调用torch模块的部分其实还相对简单，因为容易出错的地方都是被封装好了，你可以**调用模块来设计网络模型（仅限于你学过的），设置损失函数与超参数**，并在这个过程里体会到神经网络的魅力。

#### step3 我们的具体目标

第一步当然是复现李沐《手动学习神经网络》的代码。

第二步更改代码的模型、超参数（epoch、lr、dropout等）

第三步更改数据预处理方法





### 具体实现

#### 获得数据集

我们定义字典DATA_HUB，把数据集名称的字符串映射到一个二元组上（它的索引是数据集的名字），二元组内容是数据集 的url和验证数据集完整性的sha-1密钥。

```python
import hashlib
import os
import tarfile
import zipfile
import requests

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

我们要知道../一般是指本文件所在的目录,./是根目录，我们来定义一个函数来下载数据集并存放在(../data)中，并返回它的名称，如果缓存目录中已存在了，那么就使用现有的而不重复下载。

```python
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名，文件默认下载在../data"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"#我们的文件名要存在在data_hub里
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)#递归的创建目录
    fname = os.path.join(cache_dir, url.split('/')[-1])#按照给定的分隔符进行拆分，并返回一个字符串数组，这里取最后的文件的名字
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)#每次读取1MB
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:#验证匹配与否的，hash也就是哈希
            return fname  # 命中缓存目录，不重复下载
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)#下载数据集要用的
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

我们有时候下下来的是一个压缩文件（zip或者tar等），我们需要定义下解压缩，顺便我们定义一个以后会遇到的下载所有DATA_HUB的数据集的函数。(这两块代码用了很多os.path，我们下边进行总结)

```python
def download_extract(name,folder=None): #@save
    '''下载并解压zip/tar文件'''
    fname=download(name)
    base_dir = os.path.dirname(fname)#去掉文件名，返回目录
    data_dir,ext=os.path.splitext(fname)#分离路径（包含文件名）和文件类型并分别返回
    if ext == '.zip':#如果文件类型是zip
        fp = zipfile.Zipfile(fname,'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)#extractall是解压压缩包中所有文件至指定*文件夹*
    return os.path.join(base_dir,folder) if folder else data_dir

def download_all():#@save
    '''下载DATA_HUB中所有文件'''
    for name in DATA_HUB:
        download(name)
```

**os.path**在python里是一个常用模块，特别是和文件系统打交道时，我们这里总结几个特别常用的，你至少应该在下次见到他的时候知道他具体是干啥的。

> **os.path.abspath**，获取文件的绝对路径
>
> ```python
> >>path='submission.csv'#(带扩展名)
> os.path.abspath(path)
> <<'C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机\\submission'
> ```
>
> **os.path.basename**，获取文件名，但是不包含目录部分，**若路径是个目录，返回的就是空**
>
> ```python
> >>path = "C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机\\submission.csv"
> print(os.path.basename(path))
> <<submission
> >>path = "../data/"
> <<    #空字符串
> ```
>
> **os.path.dirname**,获取文件的目录部分，dirname+basename就是abspath
>
> ```python
> path = "C:/Users/lzjun/workspace/python_scripts/test_path.py"
> print(os.path.dirname(path))  
> >>C:/Users/lzjun/workspace/python_scripts
> ```
>
> **os.path.exists**,判断路径是否存在（可以判断目录和文件）
>
> ```python
> >>path = "../data/"
> os.path.exists(path)
> <<True
> >>path = "../data/submission"
> os.path.exists(path)
> <<False
> ```
>
> **os.path.getsize**,输入文件或者目录的路径，返回他们占用的大小（内存）
>
> ```python
> >path = "../data/"
> os.path.getsize(path)
> <4096
> >path = "C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机\\submission.csv"
> os.path.getsize(path)
> <23095
> ```
>
> **os.path.split**,split 方法会将路径切割成两部分，以最后一个斜杠作为切割点，第一部分是文件所在的目录， 第二部分文件名本身。如果传入的path是以“/”结尾，那么第二部就是空字符串
>
> ```python
> >path = "C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机\\submission.csv"
> os.path.split(path)
> <('C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机', 'submission.csv')
> >path = "C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机\\"
> os.path.split(path)
> <('C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机', '')
> ```
>
> **os.path.join**，拼接路径，比如我知道a的完整路径，我想在与a相同的同目录下创建b文件就可以用到join方法。
>
> ```python
> >a_path = "C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机\\submission.csv"
> dir = os.path.split(a_path)[0]
> os.path.join(dir, "b.py")
> <'C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机\\b.py'
> ```
>
> **os.path.isdir**,判断路径是否为目录，若不存在回报False
>
> ```python
> >a_path = "C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机\\submission.csv"
> os.path.isdir(a_path)
> os.path.isdir(os.path.splite(a_path)[0])
> <False
> True
> ```
>
> **os.path.isfile**,判断路径是否为文件，若不存在返回False
>
> ```python
> >print(os.path.isfile('C:\\Users\\裴英豪\\动手深度学习\\chapter4 多层感知机\\submission.csv'))
> a_path = "C:/Users/lzjun/workspace/python_scripts/a.py"#不存在这个文件
> print(os.path.isfile(a_path))
> <True
> False
> ```

#### 访问和读取数据集

现在我们想要获取我们的数据集，也就是调用上边的函数，同时把下载下来的数据集读入（使用pandas）。

我们先来导入包，并把数据集下载下来。

```python
%matplotlib inline
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
    
train_data = pd.read_csv(download('kaggle_house_train'))#这是个很重要的函数，我们会在下边进行讨论
test_data = pd.read_csv(download('kaggle_house_test'))
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))#对所有数据进行拼接[:, 1:-1]和[:, 1:]是一样的
```

pandas的read_csv函数在我们未来面对很多回归和一些分类问题的数据读取时都是第一步，我们来看看那他的一些常见使用方法,我在这里附上一个连接:[详解pandas的read_csv方法](https://zhuanlan.zhihu.com/p/340441922?utm_medium=social&utm_oi=27819925045248),这一个博主把很多参数列出来，能帮助理解。

pandas的concat是对两个数据进行连接合并的函数，可以沿着行或列进行操作，同时可以指定非合并轴的合并方式(如合集、交集等)。

```python
pd.concat(objs, axis=0, join='outer', ignore_index=False, 
          keys=None, levels=None, names=None, sort=False,
          verify_integrity=False, copy=True)
```

> objs: 需要连接的数据，可以是多个DataFrame或者Series，它是必传参数
>
> **axis: 连接轴的方法，默认值为0，即按行连接，追加在行后面;值为1时追加到列后面(按列连接:axis=1)**
>
> join: 合并方式，其他轴上的数据是按交集(inner)还是并集(outer)进行合并
>
> ignore_index: 是否保留原来的索引
>
> keys: 连接关系，使用传递的键作为最外层级别来构造层次结构索引，就是给每个表指定一个一级索引
>
> names: 索引的名称，包括多层索引
>
> verify_integrity: 是否检测内容重复;参数为True时，如果合并的数据与原数据包含索引相同的行，则会报错
>
> copy: 如果为False，则不要深拷贝.

#### 数据预处理

我们已经把csv文件分别读到了train_data和test_data里，我们下一步来进行简单的数据预处理。

我们首先要把全部的数据都给缩放到制定的范围上，缺失值替换为相应特征的平均值。
$$
x \leftarrow \frac{x - \mu}{\sigma},
$$
其中$\mu$和$\sigma$分别表示均值和标准差。 这样我们就把数据给进行了标准化。

```python
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index#获得是数字类型的数据的*索引*
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)#fillna填充缺失值 
```

然后我们用onehot向量去替代所有的离散值，我们调用get_dummies来完成

```python
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)#这些离散值会被以one-hot向量的方式转换
all_features.shape
```

我们刚刚为了方便操作通过pd.concat把test和train放一起了，我们下边把他俩分成两个数据,并转换成torch类型。

```python
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
```

#### 训练

我们先定义损失函数，并记下我们的数据有**多少列**，也就是参数的个数，并确定一个线性的网络模型

```python
loss = nn.MSELoss()
in_features = train_features.shape[1]#这个是我们的参数个数

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

我们的调出来的loss是均方损失，但是我们的房价损失其实更关心他的相对数量，即$\frac{y - \hat{y}}{y}$,这很容易理解，北京一套房1000w，你预测是990w，差距10w，但是你很厉害了，如果是鹤岗3w，你花了13w，这样的损失就比较不能接受。

我们可以用对数来衡量差异，即$\delta$用$|\log y - \log \hat{y}| \leq \delta$来表示。
$$
\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.
$$
我们的代码实现如：

```python
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()
```

上边使用了一个torch.clamp(),这个函数是吧输入的数据压缩到min和max之间，小于min的变成min，大于max的变成max。

我们的训练可以用下边的函数实现:

```python
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```



#### K折交叉验证

我们在前文中介绍了交叉验证，他有主意模型选择和帮助调整超参并在一定程度上弥补了数据集过小的问题。

```python
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k#每一‘折’的大小
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

当我们在K折交叉中训练K次后，返回训练和验证误差的平均值:

```python
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0#计算损失的累积
    for i in range(k):#对每一折进行训练
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```



#### 开始训练

这里我们给出一些超参数，也可以进行自己摸索:

```python
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
```

![1685607051305](C:\Users\裴英豪\Desktop\思维导图总结\第二个月\深度学习笔记\source\kaggle.png)



#### 提交kaggle预测

在最后面对kaggle系统的时候，我们被要求提交一个跟测试集大小相同的，包含ID和house_price的csv文件，所以我们要把预测的结果生成。

```python
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)#在列方向上进行评定
    submission.to_csv('submission.csv', index=False)
```

获取文件的绝对路径

