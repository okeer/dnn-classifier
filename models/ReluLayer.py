from ActivationFunctions import relu
from LayerBase import LayerBase


class ReluLayer(LayerBase):
    def forward(self, size_of_previous_layer):
        super(ReluLayer, self).forward(size_of_previous_layer)
        self.activation_layer = relu(self.pre_activation)
        return self.activation_layer
