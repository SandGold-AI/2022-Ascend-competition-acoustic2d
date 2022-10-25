## 问题描述

二维Acoustic Wave方程
$$
R_{PDE} := \alpha^2 \nabla^2 \phi - \frac{\partial^2 \phi}{\partial t^2}\\
R_{P.C} := \rho \alpha^2 \nabla^2 \phi(x, t, z=0)\\
R_{S_1} := \nabla \phi(x, z, t=t_1^0) - U_1^0(x, z)\\
R_{S_2} := \nabla \phi(x, z, t=t_2^0) - U_2^0(x, z)\\
R_{obs} := \nabla \phi(x, z, t) - U_{obs}(x, z, t)
$$
其中定义域为
$$
\{(x, z, t) | (x, z, t) \in [0, 1.2]\times[0, 0.45]\times[0, 0.5]\}.
$$

## PINNs方法求解Acoustic Wave方程

PINNs方法的优化目标为：
$$
\min_\Theta MSE(\Theta) = \lambda_1 MSE_{PDE} + \lambda_2 MSE_{S} + \lambda_3 MSE_{P.C} + \lambda_4 MSE_{Obs},
$$
其中
$$
MSE_S = MSE_{S_1} + MSE_{S_2},
$$
根据论文，系数$\lambda_1=0.1, \lambda_2=1, \lambda_3=1, \lambda_4=0.1$。

## 模型结构

采用与论文一致的8层100维MLP网络。

## 数据集

包括论文给出的两个时刻的初始快照数据，部分用于反演参数的真实数据，以及每100个epoch采样一次的40,000内部点和5,000边界点。

## 运行环境要求

计算硬件：Ascend 计算芯片

计算框架：Mindspore 1.7.0，numpy 1.21.2，matplotlib 3.5.1，scipy 1.5.4



## 代码框架

```
.
└─PINNforAcoustic2d
  ├─README.md
  ├─requirements.txt
  ├─solve.py                          # train
  ├─event1                            # seismic data
  ├─src
    ├──config.py                      # parameter configuration
    ├──dataset.py                     # dataset
    ├──model.py                       # network structure

```





## 超参数设置

```
import argparse


class OptionsWaveInverse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        parser.add_argument('--epochs_Adam', type=int, default=500000, help='epochs for Adam optimizer')
        parser.add_argument('--epochs_LBFGS', type=int, default=1000, help='epochs for LBFGS optimizer')
        parser.add_argument('--acoustic_layers', type=list, default=[3, 100, 100, 100, 100, 100, 100, 100, 100, 1], help='a list contain number of neuron a each layers of pde network')
        parser.add_argument('--wavespeed_layers', type=list, default=[2, 20, 20, 20, 20, 20, 20, 20, 1], help='a list contain number of neuron a each layers of alpha network')
        parser.add_argument('--weight_init', type=str, default='XavierUniform', help='trainable weight_init parameter')
        parser.add_argument('--res_batch', type=int, default=40000, help='batch size of residue sampling points')
        parser.add_argument('--bcs_batch', type=int, default=5000, help='batch size of boundary sampling points')
        self.parser = parser

    def parse(self):
        arg = self.parser.parse_args(args=[])
        return arg
```



## 模型训练

可以直接使用solve.py文件进行PINNs模型训练和求解Acoustic Wave方程。在训练过程中，模型的参数和训练过程也会被自动保存

```
python solve.py
```

模型的损失值会实时展示出来，变化如下：

```
start training...
it: 100, loss: 1.681e+00, time: 148
it: 200, loss: 1.690e+00, time: 18
it: 300, loss: 1.670e+00, time: 18
it: 400, loss: 1.671e+00, time: 18
it: 500, loss: 1.672e+00, time: 18
it: 600, loss: 1.693e+00, time: 18
it: 700, loss: 1.666e+00, time: 18
it: 800, loss: 1.691e+00, time: 18
it: 900, loss: 1.659e+00, time: 18
it: 1000, loss: 1.662e+00, time: 18
it: 1100, loss: 1.670e+00, time: 18
it: 1200, loss: 1.658e+00, time: 18
it: 1300, loss: 1.665e+00, time: 18
it: 1400, loss: 1.655e+00, time: 18
it: 1500, loss: 1.655e+00, time: 18
it: 1600, loss: 1.654e+00, time: 18
...
it: 499700, loss: 1.384e-03, time: 18
it: 499800, loss: 1.493e-03, time: 18
it: 499900, loss: 1.375e-03, time: 18
it: 500000, loss: 1.555e-03, time: 18
```

## MindScience官网

可以访问官网以获取更多信息：https://gitee.com/mindspore/mindscience