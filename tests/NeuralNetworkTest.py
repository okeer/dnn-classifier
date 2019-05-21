import unittest
import os
from utils.DataLoader import *


class NeuralNetworkTest(unittest.TestCase):
    def __read_test_set(self):
        self.test_features, self.test_classes = load_dataset(os.environ["DATASET_TEST_PATH"],
                                         os.environ["DATASET_FEATURES_TEST"],
                                         os.environ["DATASET_CLASSES_TEST"])

    def __read_train_set(self):
        self.train_features, self.train_classes = load_dataset(os.environ["DATASET_TRAIN_PATH"],
                                          os.environ["DATASET_FEATURES"],
                                          os.environ["DATASET_CLASSES"])

    def setUp(self):
        self.__read_train_set()
        self.__read_test_set()

    def test_givenDataset_whenPredict_shouldReturnClasses(self):
        nn = NeuralNetwork()

        nn.train(self.train_features, self.train_classes)

        predictions = nn.predict(self.train_features)

        self.assertEqual(predictions.shape[1], self.train_features.shape[1])


if __name__ == '__main__':
    unittest.main()
