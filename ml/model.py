"""Model architecture for the MNIST digit classifier (ANN)."""

from keras.layers import Dense, Flatten
from keras.models import Sequential

from ml.config import HIDDEN_UNITS_1, HIDDEN_UNITS_2, IMAGE_SIZE, LOSS, NUM_CLASSES, OPTIMIZER


def build_model():
    """Build the same Flatten -> Dense -> Dense -> Dense ANN as the notebook."""
    model = Sequential()
    model.add(Flatten(input_shape=IMAGE_SIZE))
    model.add(Dense(units=HIDDEN_UNITS_1, activation="relu"))
    model.add(Dense(units=HIDDEN_UNITS_2, activation="relu"))
    model.add(Dense(units=NUM_CLASSES, activation="softmax"))

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=["sparse_categorical_accuracy"])
    return model
