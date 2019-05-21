import numpy as np


def sigmoid(pre_activation):
    return 1/(1+np.exp(-pre_activation))


def relu(pre_activation):
    return (pre_activation > 0.5).astype(int)
