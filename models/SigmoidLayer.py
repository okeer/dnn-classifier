from ActivationFunctions import sigmoid
from LayerBase import LayerBase


class SigmoidLayer(LayerBase):
    def forward(self, size_of_previous_layer):
        super(SigmoidLayer, self).forward(size_of_previous_layer)
        self.activation = sigmoid(self.pre_activation)
        return self.activation
