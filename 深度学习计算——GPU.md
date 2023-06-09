## GPU

你应该确认你的电脑GPU是nvidia的，下载好了CUDA并设置适当的路径，接下来可以调用下边的代码来看到你的显卡的信息。

```python
!nvidia-smi
```

```python
Sat Jun  3 16:09:20 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 531.14                 Driver Version: 531.14       CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                      TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1050 Ti    WDDM | 00000000:01:00.0 Off |                  N/A |
| N/A   52C    P8               N/A /  N/A|      0MiB /  4096MiB |      1%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```

> 在PyTorch中，每个数组都有一个设备（device）， 我们通常将其称为环境（context）。 默认情况下，所有变量和相关的计算都分配给CPU。 有时环境可能是GPU。 当我们跨多个服务器部署作业时，事情会变得更加棘手。 通过智能地将数组分配给环境， 我们可以最大限度地减少在设备之间传输数据的时间。 例如，当在带有GPU的服务器上训练神经网络时， 我们通常希望模型的参数在GPU上。
>
> 要运行此部分中的程序，至少需要两个GPU。 注意，对大多数桌面计算机来说，这可能是奢侈的，但在云中很容易获得。 例如可以使用AWS EC2的多GPU实例。 本书的其他章节大都不需要多个GPU， 而本节只是为了展示数据如何在不同的设备之间传递。

李沐老师的这部分的部分代码需要两个GPU，如果你没有的话（当然我也没有），了解如何把数组放在GPU而不是CPU上，并领略现代计算的魅力既可。

### 计算设备

我们可以制定存储和计算的设备，如GPU和CPU。**默认情况下，tensor在内存中创建，并使用CPU计算之。**

pytorch使用torch.device(‘cpu’)和torch.device(‘cuda’)来分别表示cpu和gpu。cpu意味着所有的物理内存和cpu，但是gpu只代表一张卡和它自己带的显存。若有多个

```python
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
```

```python
(device(type='cpu'), device(type='cuda'), device(type='cuda', index=1))
```

我们也可以查询可用GPU的数量。

```python
torch.cuda.device_count()
```

```python
1
```

我电脑上只有一个1050ti所以是1

现在我们定义了两个方便的函数， 这两个函数允许我们在不存在所需所有GPU的情况下运行代码。

```python
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```

  ```python
(device(type='cuda', index=0),
 device(type='cpu'),
 [device(type='cuda', index=0)])
  ```



### 张量和GPU

我们可以查询张量所在的设备。默认情况下，张量是在CPU上创建的

```python
x = torch.tensor([1, 2, 3])
x.device
```

```python
device(type='cpu')
```

**需要注意的是，无论何时我们要对多个项进行操作， 它们都必须在同一个设备上。 例如，如果我们对两个张量求和， 我们需要确保两个张量都位于同一个设备上， 否则框架将不知道在哪里存储结果，甚至不知道在哪里执行计算。**  

#### 存储在GPU上

有几种方法可以在GPU上存储张量。 例如，我们可以在创建张量时指定存储设备。接 下来，我们在第一个`gpu`上创建张量变量`X`。 在GPU上创建的张量只消耗这个GPU的显存。 我们可以使用`nvidia-smi`命令查看显存使用情况。 一般来说，我们需要确保不创建超过GPU显存限制的数据。

```python
X = torch.ones(2, 3, device=try_gpu())
X
```

```python
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
```

假设我们有两个GPU（我没有），下边的代码可以在第二个GPU上创建一个随机张量。

```python
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```python
tensor([[0.3821, 0.5270, 0.4919],
        [0.9391, 0.0660, 0.6468]], device='cuda:1')
```

#### 复制到另一个CPU里

如果我们要计算`X + Y`，我们需要决定在哪里执行这个操作。如下图所示，我们可以将`X`传输到第二个GPU并在那里执行操作。 *不要*简单地`X`加上`Y`，因为这会导致异常， 运行时引擎不知道该怎么做：它在同一设备上找不到数据会导致失败。 由于`Y`位于第二个GPU上，所以我们需要将`X`移到那里， 然后才能执行相加运算。

![../_images/copyto.svg](C:\Users\裴英豪\Desktop\思维导图总结\第二个月\深度学习笔记\source\copyto.svg)

```python
Z = X.cuda(1)
print(X)
print(Z)
```





### 神经网络与GPU

类似地，神经网络模型可以指定设备。 下面的代码将模型参数放在GPU上。

```python
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

  

当输入为GPU上的张量时，模型将在同一GPU上计算结果。

```python
X = torch.ones(2, 3, device=try_gpu())
net(X)
```

```python
tensor([[-0.8843],
        [-0.8843]], device='cuda:0', grad_fn=<AddmmBackward0>)
```

总之，只要所有的数据和参数都在同一个设备上， 我们就可以有效地学习模型。 在以后的学习中，我们将看到几个这样的例子。


  ### 总结

- 我们可以指定用于存储和计算的设备，例如CPU或GPU。默认情况下，数据在主内存中创建，然后使用CPU进行计算。
- 深度学习框架要求计算的所有输入数据都在同一设备上，无论是CPU还是GPU。
- 不经意地移动数据可能会显著降低性能。一个典型的错误如下：计算GPU上每个小批量的损失，并在命令行中将其报告给用户（或将其记录在NumPy `ndarray`中）时，将触发全局解释器锁，从而使所有GPU阻塞。最好是为GPU内部的日志分配内存，并且只移动较大的日志。



在选择GPU或者CPU的时候，我们可以通过一个程序来看一下他们的对比:

```python
import time
import torch

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

startTime1=time.time()
for i in range(100):
    A = torch.ones(500,500)
    B = torch.ones(500,500)
    C = torch.matmul(A,B)
endTime1=time.time()

startTime2=time.time()
for i in range(100):
    A = torch.ones(500,500,device=try_gpu())
    B = torch.ones(500,500,device=try_gpu())
    C = torch.matmul(A,B)
endTime2=time.time()

print('cpu计算总时长:', round((endTime1 - startTime1)*1000, 2),'ms')
print('gpu计算总时长:', round((endTime2 - startTime2)*1000, 2),'ms')
```

```python
cpu计算总时长: 170.9 ms
gpu计算总时长: 24.93 ms
```

如此我们可以明白，cpu和gpu在大量计算时，gpu具有明显优势。