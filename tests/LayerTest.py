import unittest

from layers.LayerBase import LayerBase
from utils.ActivationFunctions import *


class LayerTest(unittest.TestCase):
    def __init_test_layer(self, cls, func):
        self.layer = cls(2, func)
        self.layer.weights = np.array([[0.5, 0.3], [0.1, 0.2]])
        self.layer.bias = np.array([[0.1, 0.2]]).reshape((2, 1))
        self.features = np.array([[1, 2, 3], [2, 1, 0.5]])

    def test_givenBaseLayer_whenForward_shouldCalculatePreActivation(self):
        self.__init_test_layer(LayerBase, ActivationFunc)
        pre_activation_assrt = np.array([[1.2, 1.4, 1.75], [0.7, 0.6, 0.6]])

        self.layer.activation = self.features
        self.layer._LayerBase__forward_linear()

        self.assertEqual(True, isinstance(self.layer.pre_activation, np.ndarray))
        np.testing.assert_array_almost_equal(pre_activation_assrt, self.layer.pre_activation)

    def test_givenSigmoidLayer_whenForward_shouldCalculateActivation(self):
        self.__init_test_layer(LayerBase, SigmoidFunc)
        sigmoid_activation_assert = np.array([[0.76852478, 0.80218389, 0.8519528],
                                              [0.66818777, 0.64565631, 0.64565631]])

        activation = self.layer.forward(self.features)

        np.testing.assert_array_almost_equal(sigmoid_activation_assert, activation)

    def test_givenReluLayer_whenForward_shouldCalculateActivation(self):
        self.__init_test_layer(LayerBase, ReluFunc)
        relu_activation_assert = np.array([[1.2, 1.4, 1.75],
                                           [0.7, 0.6, 0.6]])

        activation = self.layer.forward(self.features)

        np.testing.assert_array_almost_equal(relu_activation_assert, activation)

    def test_givenSigmoidLayer_whenBackward_shouldCalculateSlope(self):
        self.layer = LayerBase(2, SigmoidFunc)
        np.random.seed(1)
        d_activation = np.random.randn(1, 2)
        self.layer.activation = np.random.randn(3, 2)
        self.layer.weights = np.random.randn(1, 3)
        self.layer.bias = np.random.randn(1, 1)
        self.layer.pre_activation = np.random.randn(1, 2)
        d_activation_from_prev = np.array([[0.12624794, -0.04703764],
                                           [-0.09867912, 0.03676601],
                                           [0.57857522, -0.2155664]])

        np.testing.assert_array_almost_equal(d_activation_from_prev, self.layer.backward(d_activation))

    def test_givenReluLayer_whenBackward_ShouldCalculateSlope(self):
        self.layer = LayerBase(2, ReluFunc)
        np.random.seed(2)
        d_activation = np.random.randn(1, 2)
        self.layer.activation = np.random.randn(3, 2)
        self.layer.weights = np.random.randn(1, 3)
        self.layer.bias = np.random.randn(1, 1)
        self.layer.pre_activation = np.random.randn(1, 2)
        d_activation_from_prev_vrf = np.array([[0.44090989, -0.],
                                               [0.37883606, -0.],
                                               [-0.2298228, 0.]])

        d_activation_from_prev = self.layer.backward(d_activation)

        np.testing.assert_array_almost_equal(d_activation_from_prev_vrf, d_activation_from_prev)


if __name__ == '__main__':
    unittest.main()
