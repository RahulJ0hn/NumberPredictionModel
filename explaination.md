# Training Process Explained

This document walks through how the digit-recognition model is built and trained, step
by step, and maps each step back to the file in `ml/` that implements it. It's a
modular rewrite of the original notebook (`MNIST Digits Classification [0 to 9].ipynb`) —
same architecture, same hyperparameters, just split into reusable Python modules
instead of notebook cells.

## The pipeline at a glance

```
ml/data.py    -> load + normalize MNIST
ml/model.py   -> define the network architecture
ml/train.py   -> fit the model and save it to disk
ml/predict.py -> preprocess new images and run inference
```

Running `python -m ml.train` executes the whole training process end to end.

---

## Step 1: Data collection (`ml/data.py`)

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

MNIST is a dataset of 70,000 handwritten digit images (0–9), each 28×28 pixels,
grayscale. Keras downloads and caches it automatically the first time this runs.

- `x_train`: 60,000 training images, shape `(60000, 28, 28)`
- `y_train`: 60,000 labels (the correct digit 0–9 for each image)
- `x_test` / `y_test`: 10,000 held-out images/labels used only for evaluation —
  the model never trains on these, so they measure how well it generalizes.

The `assert` statements in `load_mnist()` simply guard against Keras ever changing
the dataset shape underneath us — if they fail, something's wrong before training
even starts.

## Step 2: Data preparation / normalization (`ml/data.py`)

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

Raw pixel values are integers from 0 (black) to 255 (white). Neural networks train
faster and more reliably on small, consistent-scale numbers, so every pixel is
divided by 255 to compress the range down to `[0.0, 1.0]`. This is the only
preprocessing the *training* data needs — MNIST images already come pre-centered
and pre-cropped to a consistent digit size within the 28×28 frame.

## Step 3: Model architecture (`ml/model.py`)

```python
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=100, activation="relu"))
model.add(Dense(units=30, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
```

This is a plain **Artificial Neural Network (ANN)** — fully-connected (dense) layers
only, no convolution. Layer by layer:

1. **`Flatten`** — turns each 28×28 image into a single flat vector of 784 numbers.
   Dense layers only understand 1D vectors, not 2D grids, so this is just a
   reshape, not a learned operation.
2. **`Dense(100, relu)`** — the first hidden layer. Each of its 100 neurons looks
   at all 784 input pixels and learns a weighted combination of them. `relu`
   (Rectified Linear Unit) is the activation function — it zeroes out negative
   values, which is what lets the network learn non-linear patterns instead of
   just a straight line/plane through the data.
3. **`Dense(30, relu)`** — a second, smaller hidden layer that recombines the
   100 features from the previous layer into 30 higher-level features. Stacking
   layers like this lets the network build up progressively more abstract
   representations of "what a digit looks like."
4. **`Dense(10, softmax)`** — the output layer. One neuron per digit class
   (0–9). `softmax` converts the raw outputs into a probability distribution
   that sums to 1 — e.g. `[0.01, 0.00, 0.97, ...]` means "97% sure this is a 2."

Note: because this network has no convolutional layers, it has no built-in
*translation invariance* — it's sensitive to exactly where/how pixels are
positioned, not just what shape they form. That's why `ml/predict.py` has to
carefully re-center hand-drawn input to match MNIST's own centering convention
(see the "Inference preprocessing" section below) — a shift of even a couple
pixels can measurably change the prediction.

## Step 4: Compiling the model (`ml/model.py`)

```python
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", ...)
```

Compiling wires up *how* the network learns:

- **Optimizer (`rmsprop`)** — the algorithm that adjusts the network's weights
  after each batch to reduce error. RMSprop adapts the learning rate per
  parameter based on recent gradient history, which tends to train faster and
  more stably than plain gradient descent.
- **Loss (`sparse_categorical_crossentropy`)** — the function that measures
  "how wrong" a prediction is. `sparse_` means the labels are plain integers
  (`4`, not a one-hot vector like `[0,0,0,0,1,0,0,0,0,0]`), which saves having
  to convert labels ahead of time. Cross-entropy specifically penalizes
  confident-but-wrong predictions much more heavily than unsure ones.
- **Metric (`sparse_categorical_accuracy`)** — just for human-readable
  monitoring during training; it doesn't affect how the model learns.

## Step 5: Training (`ml/train.py`)

```python
model.fit(
    x=x_train, y=y_train,
    batch_size=32, epochs=10,
    validation_data=(x_test, y_test),
)
```

- **Epoch** — one full pass through all 60,000 training images.
- **Batch size (32)** — the network doesn't look at all 60,000 images at once;
  it processes 32 at a time, computes the average error for that batch, and
  updates its weights. Smaller batches mean more frequent (but noisier) weight
  updates; 32 is a common, well-tested default.
- **`validation_data`** — after *every* epoch, the model is evaluated against
  the 10,000 test images it never trains on. This is how you catch overfitting:
  if training accuracy keeps climbing but validation accuracy stalls or drops,
  the model is memorizing the training set rather than generalizing.

Over 10 epochs on this run, training accuracy reached ~99.1% while validation
accuracy settled at ~97.6% — the small gap between the two is normal and
expected for a model this size on MNIST.

## Step 6: Saving the model (`ml/train.py`)

```python
model.save(MODEL_PATH)  # models/digits_intel.keras
```

Training from scratch takes a few minutes; saving the trained weights and
architecture to a single `.keras` file means the server can load an
already-trained model instantly at startup instead of retraining every time.

## Step 7: Inference preprocessing (`ml/predict.py`)

Training data arrives pre-normalized by MNIST's own creators — every digit
already sits centered in its 28×28 frame with a consistent size and margin.
A hand-drawn canvas image has none of those guarantees (the user might draw a
tiny digit in a corner, or a huge one filling the whole canvas), so before
handing a drawing to the model, `prepare_image()` reconstructs that same
normalization:

1. **Auto-invert** — MNIST digits are light strokes on a dark background;
   canvas drawings are the opposite by default, so colors are flipped when
   needed.
2. **Crop to the ink's bounding box** — finds just the drawn pixels and
   discards the surrounding empty canvas.
3. **Scale preserving aspect ratio** — resizes so the digit's longer side
   becomes 20px (MNIST digits occupy roughly 20×20 of their 28×28 frame),
   without squashing a narrow digit like "1" into a square.
4. **Center by center of mass** — rather than centering the bounding box
   geometrically, the digit is shifted so its pixel-intensity-weighted center
   of mass lands on the frame's center — this matches how the original MNIST
   dataset itself was normalized, which matters a lot given the network's
   sensitivity to exact pixel position (see Step 3).

This preprocessing was tuned by testing against real MNIST test images to
confirm it doesn't regress the model's ~97.6% baseline accuracy, while also
correctly classifying synthetic off-center/odd-scale digits that simulate
sloppy canvas drawings.

---

## Known limitation

This model classifies exactly **one digit per image** — it has no concept of
multi-digit numbers. Drawing "73" on the canvas gets squashed into one 28×28
tile and classified as a single 0–9 class, which will produce a wrong/nonsense
result. Supporting multi-digit numbers would require a segmentation step
(splitting the canvas into separate digit blobs before classifying each one) —
the underlying single-digit model itself wouldn't need to change.
