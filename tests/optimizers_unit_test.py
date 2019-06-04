import unittest

from utils.ActivationFunctions import *
from utils.optimizers import BaseOptimizer


class BaseOptimizerUnitTest(unittest.TestCase):
    def test_givenParamsSlope_whenComputeUpdate_shouldReturnParamsDelta(self):
        opt = BaseOptimizer(learning_rate=0.09)
        weights = np.array([[0.5, 0.3], [0.1, 0.2]])
        bias = np.array([[0.1, 0.2]]).reshape((2, 1))
        d_weights = np.array([[0.005, 0.003], [0.001, 0.002]])
        d_bias = np.array([[0.05], [0.03]])
        result_weight = np.array([[0.49955, 0.29973],
                                  [0.09991, 0.19982]])
        result_bias = np.array([[0.0955],
                                [0.1973]])

        delta_weights, delta_bias = opt.compute_update(None, (d_weights, d_bias))

        weights -= delta_weights
        bias -= delta_bias

        np.testing.assert_array_almost_equal(result_weight, weights)
        np.testing.assert_array_almost_equal(result_bias, bias)
