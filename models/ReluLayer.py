import numpy as np

from ActivationFunctions import relu
from LayerBase import LayerBase


class ReluLayer(LayerBase):
    def forward(self, size_of_previous_layer):
        super(ReluLayer, self).forward(size_of_previous_layer)
        self.activation_layer = relu(self.pre_activation)
        return self.activation_layer

    def backward(self, d_activation):
        d_pre_activation = np.array(d_activation, copy=True)
        d_pre_activation[self.pre_activation <= 0] = 0

        super(ReluLayer, self).backward(d_pre_activation)

        return self.d_activation
