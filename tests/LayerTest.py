import unittest
import numpy as np

from models.Layer import Layer


class LayerTest(unittest.TestCase):
    def test_givenNewLayer_whenForward_shouldCalculatePreActivation(self):
        layer = Layer(2, lambda x: x)
        layer.weights = np.array([[0.5, 0.3], [0.1, 0.2]])
        layer.bias = np.array([[0.1, 0.2]]).reshape((2, 1))
        features = np.array([[1, 2, 3], [2, 1, 0.5]])

        pre_activation_assrt = np.array([[1.2, 1.4, 1.75], [0.7, 0.6, 0.6]])

        layer.forward(features)

        self.assertEqual(True, isinstance(layer.pre_activation, np.ndarray))
        np.testing.assert_array_almost_equal(pre_activation_assrt, layer.pre_activation)


if __name__ == '__main__':
    unittest.main()
