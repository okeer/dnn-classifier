import os
import unittest

from utils.DataLoader import *


class UtilsTest(unittest.TestCase):
    def test_givenDataset_whenLoad_shouldReturnFeaturesAndClasses(self):
        features, classes = load_dataset(os.environ["DATASET_FILE_PATH"],
                                         (os.environ["DATASET_FEATURES"], os.environ["DATASET_CLASSES"]))

        self.assertEqual(isinstance(features, np.ndarray), True)
        self.assertEqual(isinstance(classes, np.ndarray), True)


if __name__ == '__main__':
    unittest.main()
