import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp


class MLP(nn.Cell):
    def __init__(self, layers_list, weight_init):
        super(MLP, self).__init__()
        self.model = nn.SequentialCell([])
        for i in range(1, len(layers_list) - 1):
            self.model.append(nn.Dense(in_channels=layers_list[i - 1],
                                       out_channels=layers_list[i],
                                       weight_init=weight_init,
                                       has_bias=True))
            self.model.append(nn.Tanh())
        self.model.append(nn.Dense(layers_list[-2], layers_list[-1],
                                   weight_init=weight_init, has_bias=True))

    def construct(self, *inputs):
        X = mnp.concatenate(inputs, axis=1)
        return self.model(X)
