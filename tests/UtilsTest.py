import unittest
import numpy as np


class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = DataLoader()

    def test_givenDataset_whenLoad_shouldReturnFeaturesAndClasses(self):
        features, classes = self.loader.load_dataset()

        self.assertEqual(isinstance(features, np.array), True)
        self.assertEqual(isinstance(classes, np.array), True)


if __name__ == '__main__':
    unittest.main()