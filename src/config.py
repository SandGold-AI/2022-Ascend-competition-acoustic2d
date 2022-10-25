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


