import os
import unittest

from numdl.utils.data import *


class UtilsTest(unittest.TestCase):
    def test_givenDataset_whenLoad_shouldReturnFeaturesAndClasses(self):
        features, classes = load_dataset(os.environ["DATASET_FILE_PATH"],
                                         (os.environ["DATASET_FEATURES"], os.environ["DATASET_CLASSES"]))

        self.assertEqual(isinstance(features, np.ndarray), True)
        self.assertEqual(isinstance(classes, np.ndarray), True)

    def test_givenImage_whenConvertingImageToNP_shouldReturnValidFeatureVector(self):
        filename = os.environ['cat_image']
        print(filename)
        with open(filename, 'rb') as image_file:
            image_array = image_to_np_array(image_file, 300, 300)
            self.assertEqual((270000, 1), image_array.shape)


if __name__ == '__main__':
    unittest.main()
