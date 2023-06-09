# California House Prices

[Predict California sales prices](https://www.kaggle.com/competitions/california-house-prices/rules)

在上边这个地址你可以看到更多的细节。

### 前言

经过前段时间的总结，我发现我还未从数据下载开始获取自己的一个项目，所以我通过一个类似于前两天训练的项目，

我觉得我需要了解数据的读取，数据的传递。每个模块我可以参考相关的知识，但是不能看着一个成品的项目，在项目结束后，我应该对各个环节有自己的认知，以应对以后面对的更为复杂的数据和模型。



### 数据集的下载

在上边的链接里，有一个Data的页面，往下滑能看到:

![1685799207104](C:\Users\裴英豪\Desktop\思维导图总结\第二个月\深度学习笔记\source\predicting California house price.png)

点击右下角的Download All，一般来说会下载到user\download的这个文件夹里。

我们把他们解压缩到我们的项目文件（.py）结尾的..\Data这个文件夹里。如下图所示：

![1685799408700](C:\Users\裴英豪\Desktop\思维导图总结\第二个月\深度学习笔记\source\predicting California house price data.png)



### 读入数据集

我们首先导入必须的一些模块

```python
import pandas as pd
import numpy as np
import torch
import os
```

然后使用os和pandas把他们读入

```python 
folder_path = './Data'
train_file=pd.read_csv(os.path.join(folder_path,'train.csv'))
test_file=pd.read_csv(os.path.join(folder_path,'test.csv'))
submission_file=pd.read_csv(os.path.join(folder_path,'sample_submission.csv'))
```

我们可以看一下这些数据长啥样

![1685800247750](C:\Users\裴英豪\Desktop\思维导图总结\第二个月\深度学习笔记\source\datalike0.png)

显然不能直接转换为tensor，所以我们需要做一些清洗。

我们先把两个数据进行一下整合，方便我们进行均一化。

```python
all_features = pd.concat((train_file.loc[:,train_file.columns!='Sold Price'],test_file.iloc[:,1:]))#loc可以对列的名进行比较，而iloc只能对应数字，但是如果列的编号是数字，要注意编号的顺序，因为loc不会去查看真正的位置，而是查看的匹配结果
all_features.info
```

![1685801468446](C:\Users\裴英豪\Desktop\思维导图总结\第二个月\深度学习笔记\source\all_features.png)

然后我们处理那些数字数据的缺失值，我们先找到是空数字的索引。然后再求那些不是空数字的，并求均值。

```python
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x:(x-x.mean())/(x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

然后我们看看那些离散的数据如何处理。我们尝试转换成one-hot可以不可以

```python
all_features=pd.get_dummies(all_features, dummy_na=True)
```

结果显示了Unable to allocate 5.73 GiB for an array with shape (79065, 77775) and data type uint8，因为我们的离散值太多，所以这里我就直接不要这些数据了。

