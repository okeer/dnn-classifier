import unittest
import numpy as np

from ReluLayer import ReluLayer
from SigmoidLayer import SigmoidLayer
from models.LayerBase import LayerBase


class LayerTest(unittest.TestCase):
    def test_givenNewLayer_whenForward_shouldCalculatePreActivation(self):
        layer = LayerBase(2, lambda x: x)
        layer.weights = np.array([[0.5, 0.3], [0.1, 0.2]])
        layer.bias = np.array([[0.1, 0.2]]).reshape((2, 1))
        features = np.array([[1, 2, 3], [2, 1, 0.5]])

        pre_activation_assrt = np.array([[1.2, 1.4, 1.75], [0.7, 0.6, 0.6]])

        layer._LayerBase__forward_linear(features)

        self.assertEqual(True, isinstance(layer.pre_activation, np.ndarray))
        np.testing.assert_array_almost_equal(pre_activation_assrt, layer.pre_activation)

    def test_givenNewLayer_whenForward_shouldCalculateActivation(self):
        layer = SigmoidLayer(2, lambda x: x)
        layer.weights = np.array([[0.5, 0.3], [0.1, 0.2]])
        layer.bias = np.array([[0.1, 0.2]]).reshape((2, 1))
        features = np.array([[1, 2, 3], [2, 1, 0.5]])
        sigmoid_activation_assert = np.array([[0.76852478, 0.80218389, 0.8519528],
                                              [0.66818777, 0.64565631, 0.64565631]])

        activation = layer.forward(features)

        np.testing.assert_array_almost_equal(sigmoid_activation_assert, activation)

    def test_givenReluLayer_whenForward_shouldCalculateActivation(self):
        layer = ReluLayer(2, lambda x: x)
        layer.weights = np.array([[0.5, 0.3], [0.1, 0.2]])
        layer.bias = np.array([[0.1, 0.2]]).reshape((2, 1))
        features = np.array([[1, 2, 3], [2, 1, 0.5]])
        relu_activation_assert = np.array([[1, 1, 1],
                                           [1, 1, 1]])

        activation = layer.forward(features)

        np.testing.assert_array_almost_equal(relu_activation_assert, activation)


if __name__ == '__main__':
    unittest.main()
