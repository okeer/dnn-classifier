import unittest

from utils.ActivationFunctions import *


class ActivationFuncTest(unittest.TestCase):
    def test_sigmoid(self):
        np.testing.assert_almost_equal(sigmoid(1), 0.7310585786300049)

    def test_relu(self):
        np.testing.assert_array_almost_equal(np.array([1, 1, 1, 0]), relu(np.array([0.7, 0.6, 0.9, 0.1])))


if __name__ == '__main__':
    unittest.main()
