## 图像分类数据集——fashion-Mnist

### 数据集的获取

本次实验的数据集下载自框架的内置函数torchvision.datasets中，但是在实际中可能要更加复杂。

下载的数据集需要包含一个训练集（train）和一个测试集（test）

```python
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

注意上边有个函数叫ToTensor，这是torchvision常用的把图片转换成32float函数

### 数据集的读取

将数据集放在文件源是不够的，因为目前我们进行的训练都需要从磁盘中读取到内存，而且要小批量的进行训练，所以我们需要创造一个迭代的变量，每一个epoch进行训练的时候就更新其中的值，这里我们来看其中一个实现方法。

```python
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

> 其中有个函数data.DataLodater（），他是pytorch用来处理模型输入数据的一个工具类，组合了数据集（dataset）+采样器（sampler），并在数据集上提供了单线程或多线程（num_wokers）的可迭代对象，常见的用法data.DataLoater(dataset,batch_size=1,shuffle=1,num_worker=1)
>
> 在[torch官方文档](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)中的解释如下：
>
> - **dataset** ([*Dataset*](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.Dataset)) – dataset from which to load the data.
> - **batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – how many samples per batch to load (default: `1`).
> - **shuffle** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – set to `True` to have the data reshuffled at every epoch (default: `False`).
> - **num_workers** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process. (default: `0`)

### 在实际中我们不可能每次都要从新读iter，这里我们将所有上边的组件进行封装。

```python
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]#定义将图片类型转为可运算的浮点类型
    if resize:#如果resize参数存在，就改变图片的形状
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),#返回训练集的生成
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))#返回测试集的生成
```

> 上边我们用到了transforms里的函数，这是Torch在图像领域方面常用的一个库，Resize是改变形状的，ToTensor也是它里边的。

### 我们调用封装后的函数，获得我们将会使用的两个集合train_iter, test_iter

```python
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```



## 有了训练样本之后，下边的思路就是把我们的数据放到一个函数里，去预测一个权重w和一个偏置b。

### 初始化我们的参数w和b

因为我们这里的softmax是把原始数据集中的$28\times 28$个像素点展开成了一个784长度的向量，相当于忽略掉了空间特征，我们这里只把像素位置看做一个特征，我们知道：$\hat y=\hat w\times \hat x+\hat b$而网络输出y有十个维度，所以w和b的宽度应该都是10.

```python
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

### 定义我们的softmax操作

我们再来回顾下softmax到底想干啥，softmax把一个输入的向量o，通过求幂再求商，把向量o输出成一个离散的概率分布，公式表达如：
$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.
$$
如此我们便可以写下:

```python
def softmax(X):#输入是一个（1，n）的向量
    X_exp = torch.exp(X)#对X的每一个元素都进行一个e的幂的求
    partition = X_exp.sum(1, keepdim=True)#把求幂后的数据进行加和，如果不加keepdims，最后的结果是一个标量，如果加了，最后的结果是一个（1，）的向量
    return X_exp / partition  # 这里应用了广播机制，如果不加keepdims，不同维度的数据是不能互相运算的
```

### 定义我们的模型和损失函数

对于模型来说，我们的输出应该是y，也就是$\hat y=\hat w\times \hat x+\hat b$,如果使用代码的方法写出来，那就是：

```python
y=torch.matmul(X.reshape((-1, W.shape[0])), W) + b
```

如此我们便可以定义一个softmax的函数：

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

这个损失函数的定义巧妙的利用了花式索引，我们通过一个例子来简单回顾下，什么是花式索引。

```python
y = torch.tensor([0, 2])#y代表真实的模型对应的样本标号
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])#y_hat对两个样本的三个可能性分别给出了预测的概率
y_hat[[0, 1], y]#左边的一列选择第几个预测，右边一列选择那个真实被我们预测出来的概率
>>tensor([0.1000, 0.5000])
```

如此我们可以定义交叉熵损失函数：

```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])#返回-logq（x）

cross_entropy(y_hat, y)获得p（x）log（q（x））
```

### 分类精度

在分类问题的最后，我们肯定要输出我们的判断，不能输出一个离散的分布，那么我们可以使用argmax来确定，我们的判断y_hat，这个y_hat是一个向量，我们可以与同样是向量的y进行对比y_hat==y，并进行求和，最后比上len（y），就能知道我们究竟错了多少又对了多少。代码如下：

```python
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())#返回最后预测对了的个数
```

如此我们可以简单的就把此次预测的精度计算，即accuracy(y_hat, y) / len(y)。

李沐老师在此定义了一个实用程序类，对多个变量进行累加，我们来看一下吧（老师必然是有他的深意的）

```python
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):#创建时候给n，就会返回一个（1，n）的全0向量
        self.data = [0.0] * n

    def add(self, *args):#
        self.data = [a + float(b) for a, b in zip(self.data, args)]#a是float类型的一个（1，n）向量，b是传入的参数，用来分别加给a

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

之后我们可以封装一下我们的精确度计算了：

```python
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():#这个玩意是负责关掉梯度计算，减少eval的时间
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

在上边的代码里有个Net.eval()，他的解释超过了我的理解，但是我把一个博客的东西贴上来，以后也许能看懂。

>**a) model.eval()，不启用 BatchNormalization 和 Dropout。此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。不然的话，一旦test的batch_size过小，很容易就会因BN层导致模型performance损失较大；**
>
>**b) model.train() ：启用 BatchNormalization 和 Dropout。 在模型测试阶段使用model.train() 让model变成训练模式，此时 dropout和batch normalization的操作在训练q起到防止网络过拟合的问题。**

### 训练

我们先设定一个训练一个迭代周期的函数。

```python
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):#isinstance是用来判断有一个对象是否一个已知的对象
        net.train()
    # 记录训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)#获得预测
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 如果updater已经存在在Torch里了，就使用PyTorch内置的优化器和损失函数
            updater.zero_grad()#梯度清零
            l.mean().backward()#损失函数反向求导
            updater.step()#updater是一个可定义的更新模型参数w的函数
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
```

我在这里放一个画动画的实用程序类Animator

```python
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

我们实现一个训练函数，他读取网络的名称，训练集，测试集，损失函数，迭代周期以及更新参数的方法

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)#获得训练损失和训练精度
        test_acc = evaluate_accuracy(net, test_iter)#评估在测试机上的精确度
        animator.add(epoch + 1, train_metrics + (test_acc,))#增加数据点
    train_loss, train_acc = train_metrics#获得训练集的损失和精确度
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

实际上，我们封装了那么多东西，就是为了在真正训练的时候，可以做到只调用函数，下面是一个例子：

```python
lr = 0.1#lr是学习率
def updater(batch_size):#我们定义优化算法是小批量随机梯度下降
    return d2l.sgd([W, b], lr, batch_size)
num_epochs = 10#迭代周期
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```





## softmax回归的简洁实现

### 高级的api

softmax可以变得很简单，通过使用已经做好的api。

同样的，我们先**读数据集**

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256#限定批量大小为256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

模型的参数初始化

```python
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))#sequential并非必须

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

> 这里我们用到了两个重要函数：
>
> 第一个：torch.nn.Linear(*in_features*, *out_features*, *bias=True*, *device=None*, *dtype=None*):Applies a linear transformation to the incoming data:$y=xA^T+b$
>
> *Parameters:*
>
> - **in_features** ([*int*](https://docs.python.org/3/library/functions.html#int)) – size of each input sample
> - **out_features** ([*int*](https://docs.python.org/3/library/functions.html#int)) – size of each output sample
> - **bias** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If set to `False`, the layer will not learn an additive bias. Default: `True`
>
> 第二个:torch.nn.init.normal_(*tensor*, *mean=0.0*, *std=1.0*):Fills the input Tensor with values drawn from the normal distribution$N(mean,std^2)$,相当于生成一个正态分布向量。

### 重新审视softmax的实现

在考研的计算机组成原理数据那一章，我们知道我们知道编程语言中的数值都有一个表示范围的，如果数值过大，超过最大的范围，就是上溢；如果过小，超过最小的范围，就是下溢。

softmax中间有一个步是求$\exp(x_i)$，在这一步如果$x_i$太大会出现上溢为nan，如果负的太多就会变成下溢，甚至变为0。

为了应对这种情况，我们可以使用下边的公式：
$$
\begin{aligned}
\operatorname{Softmax}\left(x_i\right) & =\frac{\exp \left(x_i\right)}{\sum_{j=1}^n \exp \left(x_j\right)} \\
& =\frac{\exp \left(x_i-b\right) \cdot \exp (b)}{\sum_{j=1}^n\left(\exp \left(x_j-b\right) \cdot \exp (b)\right)} \\
& =\frac{\exp \left(x_i-b\right) \cdot \exp (b)}{\exp (b) \cdot \sum_{j=1}^n \exp \left(x_j-b\right)} \\
& =\frac{\exp \left(x_i-b\right)}{\sum_{j=1}^n \exp \left(x_j-b\right)} \\
& =\operatorname{Softmax}\left(x_i-b\right)
\end{aligned}
$$
用到了指数归一化，避免了softmax偏离结果，调用函数的话，就可以写成：

```python
loss = nn.CrossEntropyLoss(reduction='none')
```



### 优化算法

```python
trainer=torch.optim.SGD(net.parameters(),lr=0.1)
```

>torch.optim.Optimizer(*params*, *defaults*),参数params指定了究竟是什么东西需要被优化

### 训练

```python
#调用从0的softmax的训练函数
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

