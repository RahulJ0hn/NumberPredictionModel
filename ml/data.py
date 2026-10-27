"""Data loading and preparation for the MNIST digit classifier."""

import keras

from ml.config import IMAGE_SIZE


def load_mnist():
    """Load and normalize the MNIST dataset (same as the original notebook)."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    assert x_train.shape == (60000, *IMAGE_SIZE)
    assert x_test.shape == (10000, *IMAGE_SIZE)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)
