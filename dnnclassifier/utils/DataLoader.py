import h5py
import numpy as np
from scipy.misc import imread, imresize


def load_dataset(filename, features_and_classes_names):
    train_dataset = h5py.File(filename, "r")

    features = np.array(train_dataset[features_and_classes_names[0]][:])
    features = features.reshape(features.shape[0], -1).T / 255.

    classes = np.array(train_dataset[features_and_classes_names[1]][:])
    classes = classes.reshape((1, classes.shape[0]))

    return features, classes


def image_to_np_array(file, height_pixels, width_pixels):
    image = np.array(imread(file, flatten=False))
    resized_image_array = imresize(image, size=(height_pixels, width_pixels)).reshape(
        (1, height_pixels * width_pixels * 3)).T
    return resized_image_array
