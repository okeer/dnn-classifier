import os
import pickle
import unittest

from dnnclassifier.NeuralNetwork import NeuralNetwork
from dnnclassifier.layers.LayerBase import *
from dnnclassifier.utils.ActivationFunctions import *
from dnnclassifier.utils.DataLoader import *
from dnnclassifier.utils.optimizers import BaseOptimizer, AdamOptimizer


class NeuralNetworkTest(unittest.TestCase):
    def __read_test_set(self):
        self.test_features, self.test_classes = load_dataset(os.environ["DATASET_TEST_PATH"],
                                                             (os.environ["DATASET_FEATURES_TEST"],
                                                              os.environ["DATASET_CLASSES_TEST"]))

    def __read_train_set(self):
        self.train_features, self.train_classes = load_dataset(os.environ["DATASET_FILE_PATH"],
                                                               (os.environ["DATASET_FEATURES"],
                                                                os.environ["DATASET_CLASSES"]))

    def setUp(self):
        self.__read_train_set()
        self.__read_test_set()

    def test_givenDataset_whenPredict_shouldReturnClasses(self):
        hidden = LayerBase(2, ReluFunc)
        out = LayerBase(1, SigmoidFunc)

        layers = [hidden, out]

        nn = NeuralNetwork(layers, 1, BaseOptimizer(learning_rate=0.09))

        nn.train(self.train_features, self.train_classes)

        predictions = nn.predict(self.train_features)

        self.assertEqual(predictions.shape[1], self.train_features.shape[1])

    def test__givenDataset_whenPredict_shouldMeet70PercAccuracy(self):
        layers = [LayerBase(20, ReluFunc), LayerBase(16, ReluFunc), LayerBase(7, ReluFunc), LayerBase(3, ReluFunc),
                  LayerBase(1, SigmoidFunc)]

        nn = NeuralNetwork(layers, 1000, AdamOptimizer(learning_rate=0.0075))

        nn.train(self.train_features, self.train_classes,
                 cross_validation_features=self.test_features,
                 cross_validation_classes=self.test_classes,
                 chunk_size=100)

        with open(os.environ["MODEL"], "wb") as file:
            pickle.dump(nn, file)

        predictions = nn.predict(self.train_features)

        prediction_accuracy = 100 - np.mean(np.abs(predictions - self.train_classes)) * 100
        print("Trained model Prediction accuracy is {0}%".format(prediction_accuracy))
        self.assertGreaterEqual(prediction_accuracy, 70.0)

    def test_givenFile_whenDeserialize_shouldReturnValidModel(self):
        with open(os.environ["MODEL"], "rb") as file:
            nn = pickle.load(file)

            predictions = nn.predict(self.train_features)

            prediction_accuracy = 100 - np.mean(np.abs(predictions - self.train_classes)) * 100
            print("Deserialized model Prediction accuracy is {0}%".format(prediction_accuracy))
            self.assertGreaterEqual(prediction_accuracy, 70.0)

    def test_giveSerializedModel_whenDeserialize_shouldProcessTestSetWithSufficientAccuracy(self):
        with open(os.environ["MODEL"], "rb") as file:
            nn = pickle.load(file)

            predictions = nn.predict(self.test_features)

            prediction_accuracy = 100 - np.mean(np.abs(predictions - self.test_classes)) * 100
            print("Deserialized model Prediction accuracy on test dataset is {0}%".format(prediction_accuracy))
            self.assertGreaterEqual(prediction_accuracy, 70.0)


if __name__ == '__main__':
    unittest.main()
