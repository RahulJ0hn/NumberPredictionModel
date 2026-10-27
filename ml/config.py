"""Shared configuration for the digit-prediction model pipeline."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "digits_intel.keras")

IMAGE_SIZE = (28, 28)
NUM_CLASSES = 10

HIDDEN_UNITS_1 = 100
HIDDEN_UNITS_2 = 30

BATCH_SIZE = 32
EPOCHS = 10
OPTIMIZER = "rmsprop"
LOSS = "sparse_categorical_crossentropy"
