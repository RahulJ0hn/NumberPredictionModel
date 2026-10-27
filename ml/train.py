"""Train the digit classifier and save it to disk.

Usage:
    python -m ml.train
"""

import os

from ml.config import BATCH_SIZE, EPOCHS, MODEL_PATH, MODELS_DIR
from ml.data import load_mnist
from ml.model import build_model


def train():
    (x_train, y_train), (x_test, y_test) = load_mnist()

    model = build_model()
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train()
