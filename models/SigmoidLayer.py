from ActivationFunctions import sigmoid
from LayerBase import LayerBase


class SigmoidLayer(LayerBase):
    def forward(self, size_of_previous_layer):
        super(SigmoidLayer, self).forward(size_of_previous_layer)
        self.activation_layer = sigmoid(self.pre_activation)
        return self.activation_layer

    def backward(self, d_activation):
        s = sigmoid(self.pre_activation)
        d_pre_activation = d_activation * s * (1 - s)

        super(SigmoidLayer, self).backward(d_pre_activation)

        return self.d_activation
