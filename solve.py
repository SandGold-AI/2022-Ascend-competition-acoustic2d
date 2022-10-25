import os
import numpy as np
import time
import matplotlib.pyplot as plt

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import context
from mindspore.common import set_seed

from src.model import MLP
from src.dataset import DatasetWaveInverse
from src.config import OptionsWaveInverse

import time


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class GradFirst(nn.Cell):
    def __init__(self, net):
        super(GradFirst, self).__init__()
        grad_op = ops.GradOperation(get_all=True)

        grad_first = grad_op(net)
        self.grad_first = grad_first

    def construct(self, *inputs):
        return self.grad_first(*inputs)


class GradSecond(nn.Cell):
    def __init__(self, net, num_batch):
        super(GradSecond, self).__init__()
        self.grad1 = ops.GradOperation(get_all=True, sens_param=False)
        self.first_grad = self.grad1(net)

        self.grad2 = ops.GradOperation(get_all=True, sens_param=True)
        self.second_grad = self.grad2(self.first_grad)

        self.sens1 = ms.Tensor(np.ones([num_batch, 1]).astype('float32'))
        self.sens2 = ms.Tensor(np.zeros([num_batch, 1]).astype('float32'))

    def construct(self, x, z, t):
        dxdx, dxdz, dxdt = self.second_grad(x, z, t, (self.sens1, self.sens2, self.sens2))
        dzdx, dzdz, dzdt = self.second_grad(x, z, t, (self.sens2, self.sens1, self.sens2))
        dtdx, dtdz, dtdt = self.second_grad(x, z, t, (self.sens2, self.sens2, self.sens1))
        return dxdx, dzdz, dtdt


class PINNWaveInverse(nn.Cell):
    def __init__(self, acoustic_net, wavespeed_net, acoustic_gradfn_first, 
                 acoustic_gradfn_second_res, acoustic_gradfn_second_bcs):
        super(PINNWaveInverse, self).__init__()
        self.acoustic_net = acoustic_net
        self.wavespeed_net = wavespeed_net
        self.acoustic_gradfn_first = acoustic_gradfn_first
        self.acoustic_gradfn_second_res = acoustic_gradfn_second_res
        self.acoustic_gradfn_second_bcs = acoustic_gradfn_second_bcs

        nx = 100.  # number of nodes along x axis. used here to remove the specfem's absorbing regions from PINN's computational domain
        nz = 100.

        n_abs = 10.  # # of nodes for absorbing B.C in both directions from specfem
        n_absx = n_abs  # nodes from left side of the domain
        n_absz = n_abs  # the top boundary is not absorbing

        ax_spec = 1.5  # domain size in specfem before removing absorbing regions
        az_spec = 0.5

        dx = ax_spec / nx
        dz = az_spec / nz

        self.Lx = 3.  # this is for scaling the wavespeed in the PDE via saling x coordinate
        self.Lz = 3.  # this is for scaling the wavespeed in the PDE via scaling z coordinate

        self.z_st = 0.1 - n_absz * dz  # We are removing the absorbing layer from z_st to make it with reference to PINN's coordinate
        self.z_fi = 0.45 - n_absz * dz
        self.x_st = 0.7 - n_absx * dx
        self.x_fi = 1.25 - n_absx * dx

    def construct(self, X_pde, X_init, X_S, X_BC_t, U_ini1x, U_ini1z, U_ini2x, U_ini2z, Sx, Sz):
        loss_res = self.loss_res(X_pde[:, 0:1], X_pde[:, 1:2], X_pde[:, 2:3])
        loss_ics_snapshots1, loss_ics_snapshots2 = self.loss_ics(X_init[:, 0:1], X_init[:, 1:2], X_init[:, 2:3],
                                                                 U_ini1x, U_ini1z, U_ini2x, U_ini2z)
        loss_seism = self.loss_seism(X_S[:, 0:1], X_S[:, 1:2], X_S[:, 2:3], Sx, Sz)
        loss_bcs = self.loss_bcs(X_BC_t[:, 0:1], X_BC_t[:, 1:2], X_BC_t[:, 2:3])
        loss = 10 * (0.1 * loss_res + loss_ics_snapshots1 + loss_ics_snapshots2 + loss_seism + 0.1 * loss_bcs)
        return loss

    def loss_res(self, x, z, t):
        alpha_star = 10 * self.wavespeed_net(x, z)  # 10* make the grad of wavespeed net larger
        lld = 1000.
        alpha_bound = 0.5 * (1 + mnp.tanh(lld * (z - self.z_st / self.Lz))) * 0.5 * (
                1 + mnp.tanh(lld * (-z + self.z_fi / self.Lz))) * 0.5 * (
                              1 + mnp.tanh(lld * (x - self.x_st / self.Lx))) * 0.5 * (
                              1 + mnp.tanh(lld * (-x + self.x_fi / self.Lx)))
        alpha = 3 + 2 * alpha_star * alpha_bound

        phi_xx, phi_zz, phi_tt = self.acoustic_gradfn_second_res(x, z, t)
        # scalar wave equation
        P = (1 / self.Lx) ** 2 * phi_xx + (1 / self.Lz) ** 2 * phi_zz
        res = phi_tt - alpha ** 2 * P
        return mnp.mean(res ** 2)

    def loss_ics(self, x, z, t, U_ini1x, U_ini1z, U_ini2x, U_ini2z):
        phi_x, phi_z, phi_t = self.acoustic_gradfn_first(x, z, t)

        phi_x_s1 = phi_x[:U_ini1x.shape[0], 0:1]
        phi_z_s1 = phi_z[:U_ini1x.shape[0], 0:1]

        phi_x_s2 = phi_x[U_ini1x.shape[0]:, 0:1]
        phi_z_s2 = phi_z[U_ini1x.shape[0]:, 0:1]

        loss_ics_snapshots1 = mnp.mean((phi_x_s1 - U_ini1x) ** 2) + mnp.mean((phi_z_s1 - U_ini1z) ** 2)
        loss_ics_snapshots2 = mnp.mean((phi_x_s2 - U_ini2x) ** 2) + mnp.mean((phi_z_s2 - U_ini2z) ** 2)
        return loss_ics_snapshots1, loss_ics_snapshots2

    def loss_seism(self, x, z, t, Sx, Sz):
        phi_x, phi_z, phi_t = self.acoustic_gradfn_first(x, z, t)
        return mnp.mean((phi_x - Sx) ** 2) + mnp.mean((phi_z - Sz) ** 2)

    def loss_bcs(self, x, z, t):
        phi_xx, phi_zz, phi_tt = self.acoustic_gradfn_second_bcs(x, z, t)
        P = (1 / self.Lx) ** 2 * phi_xx + (1 / self.Lz) ** 2 * phi_zz
        return mnp.mean(P ** 2)


class MyTrainOneStep(nn.Cell):
    def __init__(self, net, optimizer):
        super(MyTrainOneStep, self).__init__()
        self.net = net
        self.net.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss = self.net(*inputs)
        grads = self.grad(self.net, self.weights)(*inputs)
        self.optimizer(grads)
        return loss


def train():
    # 初始化参数器
    args = OptionsWaveInverse().parse()

    # 初始化训练数据
    dataset = DatasetWaveInverse(args.res_batch, args.bcs_batch)
    X_pde, X_init, X_S, X_BC_t, U_ini1x, U_ini1z, U_ini2x, U_ini2z, Sx, Sz = dataset.init_data()

    # 初始化网络模型
    acoustic_net = MLP(args.acoustic_layers, args.weight_init)
    wavespeed_net = MLP(args.wavespeed_layers, args.weight_init)

    # 初始化梯度算子
    acoustic_gradfn_first = GradFirst(acoustic_net)
    acoustic_gradfn_second_res = GradSecond(acoustic_net, args.res_batch)
    acoustic_gradfn_second_bcs = GradSecond(acoustic_net, args.bcs_batch)

    # 初始化PINN
    pinn = PINNWaveInverse(acoustic_net, wavespeed_net, acoustic_gradfn_first, acoustic_gradfn_second_res, acoustic_gradfn_second_bcs)
    pinn.to_float(ms.float16)

    # 初始化优化器
    optimizer = nn.Adam(pinn.trainable_params(), learning_rate=args.lr)

    # 初始化训练器
    train_net = MyTrainOneStep(pinn, optimizer)

    # 训练
    pinn.set_train(mode=True)
    with open('train_infos.txt', 'w') as f:
        pass
    print("start training...")
    last_time = time.time()
    for i in range(args.epochs_Adam):
        loss = train_net(X_pde, X_init, X_S, X_BC_t, U_ini1x, U_ini1z, U_ini2x, U_ini2z, Sx, Sz)
        if (i + 1) % 100 == 0:
            current_time = time.time()
            used_time = current_time - last_time
            infos = ('it: %.d, loss: %.3e, time: %d' % (i + 1, loss, used_time))
            with open('./train_infos.txt', 'a') as f:
                f.write(infos + '\n')
            print(infos)
            last_time = current_time
            X_pde = dataset.get_batch_res()
            X_BC_t = dataset.get_batch_bcs()
            if (i + 1) % 100000 == 0:
                ms.save_checkpoint(pinn.acoustic_net, "./acoustic_net.ckpt")
                ms.save_checkpoint(pinn.wavespeed_net, "./wavespeed_net.ckpt")


if __name__ == '__main__':
    train()
