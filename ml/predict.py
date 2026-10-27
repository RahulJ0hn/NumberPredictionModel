"""Inference helpers for the digit classifier.

Mirrors the notebook's `predict_image` step, extended to accept arbitrary
input images (e.g. a hand-drawn canvas export) instead of only pre-sized
MNIST-style files.
"""

import numpy as np
from keras.models import load_model
from PIL import Image

from ml.config import IMAGE_SIZE, MODEL_PATH

_model = None


def get_model():
    """Lazily load the trained model (single shared instance)."""
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


INK_THRESHOLD = 20
DIGIT_SIZE = 20  # MNIST digits occupy ~20x20 px centered in the 28x28 frame


def prepare_image(image: Image.Image) -> np.ndarray:
    """Convert an arbitrary PIL image into a normalized 28x28 MNIST-style array.

    MNIST digits are white strokes on a black background, scaled so their
    longest side fills ~20px and centered with a consistent margin in the
    28x28 frame -- aspect ratio is preserved (a narrow "1" stays narrow).
    Hand-drawn input (e.g. a browser canvas) is usually dark-on-light and
    arbitrarily sized/positioned, so we auto-invert and then replicate that
    same crop-scale-center pipeline rather than a blind resize.
    """
    gray = np.array(image.convert("L")).astype("float32")

    if gray.mean() > 127:
        gray = 255.0 - gray

    mask = gray > INK_THRESHOLD
    if not mask.any():
        tile = Image.fromarray(gray.astype("uint8")).resize(IMAGE_SIZE)
        return np.array(tile).astype("float32") / 255.0

    ys, xs = np.where(mask)
    crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype("uint8")

    # Scale so the longer side becomes DIGIT_SIZE, preserving aspect ratio.
    ch, cw = crop.shape
    scale = DIGIT_SIZE / max(ch, cw)
    new_h, new_w = max(1, round(ch * scale)), max(1, round(cw * scale))
    digit = np.array(Image.fromarray(crop).resize((new_w, new_h))).astype("float32")

    # Real MNIST normalization centers by center of mass, not bounding-box
    # geometry -- this plain dense ANN has no translation invariance, so even
    # a pixel or two of misalignment from geometric centering flips predictions.
    dy, dx = np.indices(digit.shape)
    total = digit.sum()
    cy, cx = (dy * digit).sum() / total, (dx * digit).sum() / total

    canvas = np.zeros(IMAGE_SIZE, dtype="float32")
    center = IMAGE_SIZE[0] / 2
    y_off = int(round(center - cy))
    x_off = int(round(center - cx))

    src_y0, src_x0 = max(0, -y_off), max(0, -x_off)
    dst_y0, dst_x0 = max(0, y_off), max(0, x_off)
    h = min(new_h - src_y0, IMAGE_SIZE[1] - dst_y0)
    w = min(new_w - src_x0, IMAGE_SIZE[0] - dst_x0)
    canvas[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w] = digit[src_y0 : src_y0 + h, src_x0 : src_x0 + w]

    return canvas / 255.0


def predict_image(image: Image.Image):
    """Predict the digit shown in `image`. Returns (digit, confidence, probabilities)."""
    array = prepare_image(image)
    batch = np.expand_dims(array, axis=0)

    model = get_model()
    probabilities = model.predict(batch, verbose=0)[0]

    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])

    return digit, confidence, probabilities.tolist()
