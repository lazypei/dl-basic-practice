## 暂退法（Dropout）

当训练神经网络的时候（多在全连接层使用），我们可以随机丢弃一部分神经元(同时丢弃与其相连接的边)来避免过拟合。其实现方法最简单的就是设置一个固定的概率$p\sub[0,1]$,对每一个神经元都以概率$p$来判定要不要保留。我们希望丢弃后的x和丢弃前的x期望相同：$\mathbf E[x'] = x$,为此，我们要求：
$$
\begin{split}\begin{aligned}
h' =
\begin{cases}
    0 & \text{ 概率为 } p \\
    \frac{h}{1-p} & \text{ 其他情况}
\end{cases}
\end{aligned}\end{split}
$$
![1684731522333](C:\Users\裴英豪\Desktop\思维导图总结\第二个月\深度学习笔记\source\Dropout-example.png)

#### 暂退法的系统实现

首先我们导入必要的包，并定义dropout（可以认为他是一个单独的层）：

```python
import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X,dropout):
    assert 0<= dropout <=1
    #在此种情况下，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    #在此种条件下，所有元素都被保留
    if dropout ==0:
        return X
    mask = (torch.rand(X.shape)>dropout).float()#生成一个E（mask）=dropout的随机向量
    return mask*X/(1.0-dropout)
```

然后我们把每一层的参数都给设定好：

```python
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

定义我们的模型，将暂退法应用于每个隐藏层的输出（在激活函数之前），**靠近输入层的地方设置较低的暂退概率**，这里第一个和第二个的隐藏层的暂退概率分别为0.2和0.5，并且**暂退法只在训练的时候使用**。

```python
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()#继承父类的init方法，等价于nn.module.__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

这里通过继承父类的方法，我们很方便的给参数给空间，目前来看是一种约定俗成的方法。下边我们来进行模型训练：

```python
num_epochs, lr, batch_size = 10, 0.5, 256#一共十代，学习率0.5批量大小256
loss = nn.CrossEntropyLoss(reduction='none')#交叉熵损失函数
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)#读取数据
trainer = torch.optim.SGD(net.parameters(), lr=lr)#随机梯度下降优化函数
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer) 
```

图像如下：

![1684729309319](C:\Users\裴英豪\Desktop\思维导图总结\第二个月\深度学习笔记\source\Dropout-1.png)







#### 下面是调用接口的简洁实现

我们只需要给每一层手动加上Dropout层，把暂退概率作为参数传给构造函数既可。

```python
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)#参数赋初值

net.apply(init_weights)#Sequential.apply(fn)对当前模块中的所有模块应用函数 fn，包括当前模块本身。
```

测试：

```python
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```





![1684729327881](C:\Users\裴英豪\Desktop\思维导图总结\第二个月\深度学习笔记\source\Dropout-2.png)

### 总结

- 丢弃法将一些输出项随机设置为0来控制模型复杂度
- 常作用在多层感知机的隐藏层输出上
- 丢弃概率是控制模型复杂度的超参
- 暂退法在前向传播过程中，计算每一内部层的同时丢弃一些神经元。
- 暂退法可以避免过拟合，它通常与控制权重向量的维数和大小结合使用的。
- 暂退法将活性值h替换为具有期望值h的随机变量。
- 暂退法仅在训练期间使用。