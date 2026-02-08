# Phase 3: Create `experiments/mnist/` Implementation Plan

> **For Agent:** REQUIRED: Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Move all MNIST-specific code from `main.py`, `mnist_rnn_model.py`, and `oar/training.py` into a proper `experiments/mnist/` directory structure.

**Architecture:** Create `experiments/mnist/` with three modules: `data.py` (dataset loading/preprocessing), `model.py` (architecture), and `config.py` (options and layer configuration). Remove MNIST-specific code from `oar/training.py`. Keep `main.py` as a thin experiment runner.

**Tech Stack:** TensorFlow 2.10.1, QKeras 0.9.0, Python 3.10, tensorflow-datasets

---

## Pre-Phase Analysis

### Code to Move

| Current Location | Target Location | Content |
|------------------|-----------------|---------|
| `main.py:61-142` | `experiments/mnist/data.py` | `normalize_img`, `resize`, `augment`, `get_datasets` |
| `mnist_rnn_model.py` (entire file) | `experiments/mnist/model.py` | `get_model`, initializers, regularizers |
| `main.py:247-283` | `experiments/mnist/config.py` | `get_model_parameter_stats` |
| `main.py:285-397` | `experiments/mnist/config.py` | `perform_step_in_four_step_quant`, `perform_four_step_quant` |
| `main.py:400-509` | `experiments/mnist/config.py` | Training entrypoints, options dicts |
| `oar/training.py:7-68` | `experiments/mnist/config.py` | `get_default_layer_options_from_options` (MNIST-specific!) |

### Files That Will Import from experiments/mnist/

- `main.py` - will import from `experiments.mnist`
- `export_mnist_weights_h5.py` - uses `get_model` and `get_default_layer_options_from_options`
- `export_mnist.py` - standalone, only uses `oar.ternarize_tensor_with_threshold`

### Invariants to Preserve

1. **Same model architecture** - parameter counts must match
2. **Same data pipeline** - train/val/test splits, preprocessing, augmentation
3. **Same training behavior** - four-step quantization produces identical results
4. **Exports work** - `export_mnist_weights_h5.py` and `export_mnist.py` still function

---

## Task 1: Create experiments/mnist/data.py

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/mnist/__init__.py`
- Create: `experiments/mnist/data.py`
- Test: `tests/test_mnist_data.py`

### Step 1: Create directory structure

Run:
```bash
mkdir -p experiments/mnist
```

### Step 2: Create experiments/__init__.py

Create `experiments/__init__.py`:

```python
"""Experiment implementations using the OAR framework."""
```

### Step 3: Create experiments/mnist/__init__.py

Create `experiments/mnist/__init__.py`:

```python
"""MNIST RNN experiment with OAR regularization."""

from experiments.mnist.data import get_datasets, normalize_img, resize, augment
from experiments.mnist.model import get_model
from experiments.mnist.config import (
    MNIST_OPTIONS,
    get_default_layer_options,
    get_model_parameter_stats,
    perform_step_in_four_step_quant,
    perform_four_step_quant,
)

__all__ = [
    "get_datasets",
    "normalize_img",
    "resize",
    "augment",
    "get_model",
    "MNIST_OPTIONS",
    "get_default_layer_options",
    "get_model_parameter_stats",
    "perform_step_in_four_step_quant",
    "perform_four_step_quant",
]
```

### Step 4: Write failing test for data module

Create `tests/test_mnist_data.py`:

```python
"""Tests for experiments/mnist/data.py."""

import tensorflow as tf


def test_normalize_img_divides_by_255():
    from experiments.mnist.data import normalize_img

    image = tf.constant([[[255.0]]], dtype=tf.float32)
    label = tf.constant(5)
    norm_img, norm_label = normalize_img(image, label)
    tf.debugging.assert_near(norm_img, tf.constant([[[1.0]]]))
    tf.debugging.assert_equal(norm_label, label)


def test_resize_to_128x128():
    from experiments.mnist.data import resize

    image = tf.zeros([28, 28, 1], dtype=tf.float32)
    label = tf.constant(3)
    resized_img, resized_label = resize(image, label)
    assert resized_img.shape == (128, 128, 1)
    tf.debugging.assert_equal(resized_label, label)


def test_get_datasets_returns_three_datasets():
    from experiments.mnist.data import get_datasets

    ds_train, ds_val, ds_test = get_datasets(batch_size=32, enlarge=False)
    assert isinstance(ds_train, tf.data.Dataset)
    assert isinstance(ds_val, tf.data.Dataset)
    assert isinstance(ds_test, tf.data.Dataset)


def test_get_datasets_batch_shape_28x28():
    from experiments.mnist.data import get_datasets

    ds_train, _, _ = get_datasets(batch_size=32, enlarge=False)
    for batch in ds_train.take(1):
        images, labels = batch
        assert images.shape == (32, 28, 28, 1)
        assert labels.shape == (32,)


def test_get_datasets_batch_shape_128x128():
    from experiments.mnist.data import get_datasets

    ds_train, _, _ = get_datasets(batch_size=32, enlarge=True)
    for batch in ds_train.take(1):
        images, labels = batch
        assert images.shape == (32, 128, 128, 1)


def test_augment_returns_same_shape():
    from experiments.mnist.data import augment

    image = tf.random.uniform([28, 28, 1], dtype=tf.float32)
    label = tf.constant(7)
    aug_image, aug_label = augment(image, label)
    assert aug_image.shape == image.shape
    tf.debugging.assert_equal(aug_label, label)


def test_normalize_img_preserves_zero():
    from experiments.mnist.data import normalize_img

    image = tf.zeros([28, 28, 1], dtype=tf.float32)
    label = tf.constant(0)
    norm_image, _ = normalize_img(image, label)
    tf.debugging.assert_near(norm_image, tf.zeros_like(norm_image))
```

### Step 5: Run test to verify it fails

Run: `pytest tests/test_mnist_data.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments'`

### Step 6: Create experiments/mnist/data.py

Create `experiments/mnist/data.py`:

```python
"""MNIST dataset loading and preprocessing."""

import tensorflow as tf
import tensorflow_datasets as tfds


def normalize_img(image, label):
    """Normalizes the MNIST image by dividing by the max value.

    Args:
        image: MNIST image tensor
        label: MNIST image label

    Returns:
        Tuple of (normalized image, label)
    """
    return tf.cast(image, tf.float32) / 255.0, label


def resize(image, label):
    """Resizes the image to [128, 128].

    Args:
        image: Image tensor
        label: Label tensor

    Returns:
        Tuple of (resized image, label)
    """
    return tf.image.resize(image, [128, 128]), label


def augment(image, label):
    """Applies data augmentation: random brightness and horizontal flip.

    Args:
        image: Image tensor
        label: Label tensor

    Returns:
        Tuple of (augmented image, label)
    """
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)
    return image, label


def _prepare_dataset(
    ds: tf.data.Dataset,
    enlarge: bool,
    batch_size: int,
    shuffle_buffer: int | None = None,
    apply_augment: bool = False,
) -> tf.data.Dataset:
    """Applies preprocessing pipeline to a dataset.

    Args:
        ds: Input dataset
        enlarge: If True, resize images to 128x128
        batch_size: Batch size
        shuffle_buffer: If set, shuffle with this buffer size
        apply_augment: If True, apply data augmentation

    Returns:
        Preprocessed dataset
    """
    autotune = tf.data.experimental.AUTOTUNE
    ds = ds.map(normalize_img, num_parallel_calls=autotune)
    if enlarge:
        ds = ds.map(resize, num_parallel_calls=autotune)
    ds = ds.cache()
    if shuffle_buffer is not None:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    if apply_augment:
        ds = ds.map(augment, num_parallel_calls=autotune)
    return ds.batch(batch_size, drop_remainder=True).prefetch(autotune)


def get_datasets(
    batch_size: int = 512, enlarge: bool = False
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Downloads, preprocesses, and prepares the MNIST dataset.

    Creates three dataloaders: training, validation, and test.
    - Validation set: 2500 samples from training data
    - Training set: 57500 samples
    - Test set: 10000 samples

    Optionally resizes from [28,28] to [128,128] for enlarged MNIST RNN.

    Args:
        batch_size: Batch size for training. Defaults to 512.
        enlarge: If True, resize images to 128x128. Defaults to False.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    ds, ds_test = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=False,
        as_supervised=True,
        with_info=False,
    )
    ds_val = ds.take(2500)
    ds_train = ds.skip(2500)

    ds_train = _prepare_dataset(
        ds_train, enlarge, batch_size,
        shuffle_buffer=len(ds_train), apply_augment=True
    )
    ds_val = _prepare_dataset(ds_val, enlarge, batch_size)
    ds_test = _prepare_dataset(ds_test, enlarge, batch_size)

    return ds_train, ds_val, ds_test
```

### Step 7: Run test to verify it passes

Run: `pytest tests/test_mnist_data.py -v`

Expected: All 7 tests PASS

### Step 8: Commit

```bash
git add experiments/ tests/test_mnist_data.py
git commit -m "feat(mnist): add experiments/mnist/data.py with dataset loading"
```

---

## Task 2: Create experiments/mnist/model.py

**Files:**
- Create: `experiments/mnist/model.py`
- Test: `tests/test_mnist_model.py`

### Step 1: Write failing test for model module

Create `tests/test_mnist_model.py`:

```python
"""Tests for experiments/mnist/model.py."""

import pytest
import tensorflow as tf


@pytest.fixture
def default_options():
    """Default options for model tests."""
    return {
        "enlarge": False,
        "batch_size": 32,
        "quantize": False,
        "oar": {"oar_lambda": 1e-4, "omega": 6},
        "tᵢ": 0.7,
    }


@pytest.fixture
def default_layer_options():
    """Default layer options for model tests."""
    def make_layer_config(activation, oar_lambda=1e-4):
        return {
            "activation": activation,
            "oar": {"use": False, "oar_lambda": oar_lambda, "omega": 6},
            "s": 1.0,
            "τ": 0.0,
        }

    return {
        "INPUT": {"ternarize": False},
        "QRNN_0": make_layer_config(tf.keras.activations.tanh),
        "QRNN_1": make_layer_config(tf.keras.activations.tanh),
        "DENSE_0": make_layer_config(tf.keras.activations.tanh),
        "DENSE_OUT": make_layer_config(tf.keras.activations.softmax, oar_lambda=0.0),
    }


def test_get_model_returns_keras_model(default_options, default_layer_options):
    from experiments.mnist.model import get_model

    model = get_model(default_options, default_layer_options)
    assert isinstance(model, tf.keras.Model)


def test_get_model_has_correct_input_shape_28x28(default_options, default_layer_options):
    from experiments.mnist.model import get_model

    model = get_model(default_options, default_layer_options)
    assert model.input_shape == (None, 28, 28, 1)


def test_get_model_has_correct_output_shape(default_options, default_layer_options):
    from experiments.mnist.model import get_model

    model = get_model(default_options, default_layer_options)
    assert model.output_shape == (None, 10)


def test_get_model_layer_names(default_options, default_layer_options):
    from experiments.mnist.model import get_model

    model = get_model(default_options, default_layer_options)
    layer_names = [layer.name for layer in model.layers]
    assert "QRNN_0" in layer_names
    assert "QRNN_1" in layer_names
    assert "DENSE_0" in layer_names
    assert "DENSE_OUT" in layer_names


def test_get_model_has_correct_input_shape_128x128(default_options, default_layer_options):
    from experiments.mnist.model import get_model

    default_options["enlarge"] = True
    model = get_model(default_options, default_layer_options)
    assert model.input_shape == (None, 128, 128, 1)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_mnist_model.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.mnist.model'`

### Step 3: Create experiments/mnist/model.py

Create `experiments/mnist/model.py` - copy from `mnist_rnn_model.py` with updated imports:

```python
"""MNIST RNN model architecture with OAR support."""

import tensorflow as tf
from qkeras import *

from oar import (
    Downsampling,
    QDenseWithOAR,
    QRNNWithOAR,
    OARModel,
    TrackedActivation,
    TernarizationWithThreshold,
    ternarize_tensor_with_threshold,
)

SEED = 1997

# Regularizers
kernel_regularizer = None
recurrent_regularizer = None
bias_regularizer = None
activation_regularizer = None

# Initializers
rnn_kernel_initializer = tf.keras.initializers.VarianceScaling(
    scale=1.0, mode="fan_avg", distribution="uniform", seed=SEED
)
rnn_recurrent_initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=SEED)
dense_kernel_initializer = tf.keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_in", distribution="truncated_normal", seed=SEED
)


def get_model(options, layer_options):
    """Initialize and return the MNIST RNN model.

    Args:
        options: Options dict with keys:
            - enlarge: bool, use 128x128 input (True) or 28x28 (False)
            - batch_size: int, batch size for training
            - quantize: bool, enable weight quantization
            - tᵢ: float, input ternarization threshold multiplier
        layer_options: Per-layer options dict with keys for each layer:
            INPUT, QRNN_0, QRNN_1, DENSE_0, DENSE_OUT

    Returns:
        OARModel instance
    """
    input = tf.keras.layers.Input(
        shape=(28, 28, 1) if not options["enlarge"] else (128, 128, 1)
    )

    # Ternarize inputs (if step 3+)
    input = (
        tf.keras.layers.Lambda(
            lambda x: tf.stop_gradient(
                ternarize_tensor_with_threshold(
                    x, theta=options["tᵢ"] * tf.reduce_mean(tf.abs(x))
                )
            ),
            trainable=False,
            dtype=tf.float32,
            name="TERNARIZE_WITH_THRESHOLD",
        )(input)
        if layer_options["INPUT"]["ternarize"]
        else tf.keras.layers.Lambda(lambda x: x, name="NOOP")(input)
    )

    input = tf.keras.layers.Reshape(
        target_shape=(28, 28) if not options["enlarge"] else (128, 128)
    )(input)

    qrnn_0 = QRNNWithOAR(
        cell=None,
        units=128,
        activation=TrackedActivation(
            activation=layer_options["QRNN_0"]["activation"], name="QRNN_0"
        ),
        batch_size=options["batch_size"],
        use_bias=False,
        return_sequences=True,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=(
            TernarizationWithThreshold(
                threshold=layer_options["QRNN_0"]["τ"],
                name="QRNN_0/quantized_kernel",
            )
            if options["quantize"]
            else None
        ),
        recurrent_quantizer=(
            TernarizationWithThreshold(
                threshold=layer_options["QRNN_0"]["τ"],
                name="QRNN_0/quantized_recurrent",
            )
            if options["quantize"]
            else None
        ),
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        use_oar=layer_options["QRNN_0"]["oar"]["use"],
        oar_lambda=layer_options["QRNN_0"]["oar"]["oar_lambda"],
        omega=layer_options["QRNN_0"]["oar"]["omega"],
        s=layer_options["QRNN_0"]["s"],
        name="QRNN_0",
    )(input)
    tr = Downsampling(reduction_factor=2)(qrnn_0)
    qrnn_1 = QRNNWithOAR(
        cell=None,
        units=128,
        activation=TrackedActivation(
            activation=layer_options["QRNN_1"]["activation"], name="QRNN_1"
        ),
        batch_size=options["batch_size"],
        use_bias=False,
        return_sequences=True,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=(
            TernarizationWithThreshold(
                threshold=layer_options["QRNN_1"]["τ"],
                name="QRNN_1/quantized_kernel",
            )
            if options["quantize"]
            else None
        ),
        recurrent_quantizer=(
            TernarizationWithThreshold(
                threshold=layer_options["QRNN_1"]["τ"],
                name="QRNN_1/quantized_recurrent",
            )
            if options["quantize"]
            else None
        ),
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        use_oar=layer_options["QRNN_1"]["oar"]["use"],
        oar_lambda=layer_options["QRNN_1"]["oar"]["oar_lambda"],
        omega=layer_options["QRNN_1"]["oar"]["omega"],
        s=layer_options["QRNN_1"]["s"],
        name="QRNN_1",
    )(tr)
    qrnn_1 = tf.keras.layers.Flatten()(qrnn_1)
    dense_0 = QDenseWithOAR(
        1024,
        activation=TrackedActivation(
            activation=layer_options["DENSE_0"]["activation"], name="DENSE_0"
        ),
        batch_size=options["batch_size"],
        use_bias=False,
        kernel_regularizer=kernel_regularizer,
        kernel_quantizer=(
            TernarizationWithThreshold(
                threshold=layer_options["DENSE_0"]["τ"], name="DENSE_0"
            )
            if options["quantize"]
            else None
        ),
        kernel_initializer=dense_kernel_initializer,
        use_oar=layer_options["DENSE_0"]["oar"]["use"],
        oar_lambda=layer_options["DENSE_0"]["oar"]["oar_lambda"],
        omega=layer_options["DENSE_0"]["oar"]["omega"],
        s=layer_options["DENSE_0"]["s"],
        name="DENSE_0",
    )(qrnn_1)
    output = QDenseWithOAR(
        10,
        use_bias=False,
        activation=TrackedActivation(
            activation=layer_options["DENSE_OUT"]["activation"], name="DENSE_OUT"
        ),
        batch_size=options["batch_size"],
        kernel_regularizer=kernel_regularizer,
        kernel_quantizer=(
            TernarizationWithThreshold(
                threshold=layer_options["DENSE_OUT"]["τ"], name="DENSE_OUT"
            )
            if options["quantize"]
            else None
        ),
        kernel_initializer=dense_kernel_initializer,
        use_oar=layer_options["DENSE_OUT"]["oar"]["use"],
        oar_lambda=layer_options["DENSE_OUT"]["oar"]["oar_lambda"],
        omega=layer_options["DENSE_OUT"]["oar"]["omega"],
        s=layer_options["DENSE_OUT"]["s"],
        name="DENSE_OUT",
    )(dense_0)

    model = OARModel(
        inputs=[input],
        outputs=[output],
        name="MNIST_RNN" if not options["enlarge"] else "ENLARGED_MNIST_RNN",
    )

    model.summary()

    return model
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_mnist_model.py -v`

Expected: All 5 tests PASS

### Step 5: Commit

```bash
git add experiments/mnist/model.py tests/test_mnist_model.py
git commit -m "feat(mnist): add experiments/mnist/model.py with MNIST RNN architecture"
```

---

## Task 3: Create experiments/mnist/config.py

**Files:**
- Create: `experiments/mnist/config.py`
- Modify: `oar/training.py` - remove `get_default_layer_options_from_options`
- Modify: `oar/__init__.py` - remove export
- Test: `tests/test_mnist_config.py`

### Step 1: Write failing test for config module

Create `tests/test_mnist_config.py`:

```python
"""Tests for experiments/mnist/config.py."""

import tensorflow as tf


def test_mnist_options_has_required_keys():
    from experiments.mnist.config import MNIST_OPTIONS

    assert "enlarge" in MNIST_OPTIONS
    assert "epochs" in MNIST_OPTIONS
    assert "learning_rate" in MNIST_OPTIONS
    assert "batch_size" in MNIST_OPTIONS
    assert "t" in MNIST_OPTIONS
    assert "tᵢ" in MNIST_OPTIONS
    assert "s" in MNIST_OPTIONS
    assert "oar" in MNIST_OPTIONS
    assert "quantize" in MNIST_OPTIONS


def test_get_default_layer_options_returns_all_layers():
    from experiments.mnist.config import get_default_layer_options, MNIST_OPTIONS

    layer_options = get_default_layer_options(MNIST_OPTIONS)
    assert "INPUT" in layer_options
    assert "QRNN_0" in layer_options
    assert "QRNN_1" in layer_options
    assert "DENSE_0" in layer_options
    assert "DENSE_OUT" in layer_options


def test_get_default_layer_options_uses_omega():
    from experiments.mnist.config import get_default_layer_options

    options = {
        "oar": {"omega": 8, "oar_lambda": 1e-3},
    }
    layer_options = get_default_layer_options(options)
    assert layer_options["QRNN_0"]["oar"]["omega"] == 8
    assert layer_options["QRNN_0"]["oar"]["oar_lambda"] == 1e-3


def test_get_model_parameter_stats_returns_dict():
    from experiments.mnist.config import (
        get_model_parameter_stats,
        get_default_layer_options,
        MNIST_OPTIONS,
    )

    layer_options = get_default_layer_options(MNIST_OPTIONS)
    stats = get_model_parameter_stats(None, MNIST_OPTIONS, layer_options)
    assert "QRNN_0" in stats
    assert "QRNN_1" in stats
    assert "DENSE_0" in stats
    assert "DENSE_OUT" in stats


def test_perform_step_returns_checkpoint_path():
    from experiments.mnist.config import (
        perform_step_in_four_step_quant,
        MNIST_OPTIONS,
    )
    import copy

    # Use minimal epochs for test
    options = copy.deepcopy(MNIST_OPTIONS)
    options["epochs"] = 1

    # Step 1 should return a path string
    result = perform_step_in_four_step_quant(
        step=1, pretrained_weights=None, options=options
    )
    assert isinstance(result, str)
    assert "checkpoints" in result
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_mnist_config.py::test_mnist_options_has_required_keys -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.mnist.config'`

### Step 3: Create experiments/mnist/config.py

Create `experiments/mnist/config.py`:

```python
"""MNIST experiment configuration and training orchestration."""

import tensorflow as tf
from oar import mod_sign, sign_ste_tanh

# Default options for MNIST RNN experiment
MNIST_OPTIONS = {
    "enlarge": False,
    "epochs": 100,
    "learning_rate": 1e-4,
    "batch_size": 512,
    "t": 1.5,  # Ternarization threshold multiplier
    "tᵢ": 0.7,  # Input ternarization threshold
    "s": 4.0,  # Gradient scaling factor
    "oar": {
        "oar_lambda": 1e-4,  # OAR regularization rate
        "omega": 6,  # Bit precision (2^omega modulus)
    },
    "quantize": False,
}

# Enlarged MNIST options (128x128)
ENLARGED_MNIST_OPTIONS = {
    "enlarge": True,
    "epochs": 100,
    "learning_rate": 5e-6,
    "batch_size": 512,
    "t": 1.5,
    "tᵢ": 0.7,
    "s": 4.0,
    "oar": {
        "oar_lambda": 1e-4,
        "omega": 6,
    },
    "quantize": False,
}


def get_default_layer_options(options):
    """Build default layer_options dict from high-level options.

    Creates per-layer configuration for MNIST RNN architecture.
    Used as starting point for four-step quantization.

    Args:
        options: Dict with "oar" sub-dict containing "omega" and "oar_lambda"

    Returns:
        Dict mapping layer names to their configuration
    """
    def make_oar_config(use: bool = False, oar_lambda: float | None = None):
        return {
            "use": use,
            "oar_lambda": oar_lambda if oar_lambda is not None else options["oar"]["oar_lambda"],
            "omega": options["oar"]["omega"],
        }

    result = {"INPUT": {"ternarize": False}}

    for name in ["QRNN_0", "QRNN_1", "DENSE_0"]:
        result[name] = {
            "activation": tf.keras.activations.tanh,
            "oar": make_oar_config(),
            "s": 1.0,
            "τ": 0.0,
        }

    result["DENSE_OUT"] = {
        "activation": tf.keras.activations.softmax,
        "oar": make_oar_config(use=True, oar_lambda=0.0),
        "s": 1.0,
        "τ": 0.0,
    }

    return result


def get_model_parameter_stats(pretrained_weights: str | None, options, layer_options):
    """Gets the mean of the absolute value for ternarization of each parameter.

    Used to compute per-layer ternarization thresholds: τ = t * E(|θ_l|)

    Args:
        pretrained_weights: Path to checkpoints folder containing parameters
        options: Model options
        layer_options: Layer-wise model options

    Returns:
        Dict mapping layer names to mean absolute weight values
    """
    from experiments.mnist.model import get_model

    tern_params = {"QRNN_0": 0, "QRNN_1": 0, "DENSE_0": 0, "DENSE_OUT": 0}
    model = get_model(options, layer_options)

    if pretrained_weights is None:
        return tern_params

    model.load_weights(pretrained_weights)
    print("Restored pretrained weights from {}.".format(pretrained_weights))

    # Calculate weight stats per layer
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:
            all_weights = tf.concat(
                [tf.reshape(x, shape=[-1]) for x in layer.trainable_weights], axis=-1
            )
            mean_abs = tf.math.reduce_mean(tf.abs(all_weights))
            if layer.name.find("QRNN_0") != -1:
                tern_params["QRNN_0"] = mean_abs.numpy()
            if layer.name.find("QRNN_1") != -1:
                tern_params["QRNN_1"] = mean_abs.numpy()
            if layer.name.find("DENSE_0") != -1:
                tern_params["DENSE_0"] = mean_abs.numpy()
            if layer.name.find("DENSE_OUT") != -1:
                tern_params["DENSE_OUT"] = mean_abs.numpy()
    return tern_params


def perform_step_in_four_step_quant(step: int, pretrained_weights: str | None, options) -> str:
    """Performs a step in the four-step quantization procedure.

    Steps:
    1. Train with tanh activation (baseline)
    2. Replace tanh with sign (tanh gradient), apply gradient scaling s
    3. Ternarize inputs
    4. Enable weight quantization + OAR (learning rate ×0.1)

    Args:
        step: Step number (1-4)
        pretrained_weights: Path to checkpoints folder for initialization
        options: Overall options dict

    Returns:
        Path to checkpoints folder of the trained model
    """
    # Import here to avoid circular imports
    from experiments.mnist.training import train

    print(f"\nPERFORMING STEP {step}/4 FROM FOUR-STEP QUANTIZATION PROCESS\n")

    layer_options = get_default_layer_options(options)

    # Change settings according to step number
    ternarize_inputs = False
    t = 1.0
    s = 1.0
    activation = tf.keras.activations.tanh
    oar = False
    tern_params = {"QRNN_0": 0, "QRNN_1": 0, "DENSE_0": 0, "DENSE_OUT": 0}

    if step == 2:
        s = options["s"]
        activation = sign_ste_tanh
    if step == 3:
        ternarize_inputs = True
    if step == 4:
        options["quantize"] = True
        options["learning_rate"] *= 0.1
        t = options["t"]
        oar = True
        tern_params = get_model_parameter_stats(
            pretrained_weights, options, layer_options
        )
        print("\nTERNARIZATION PARAMETERS:")
        print(tern_params)

        def activation(x):
            return mod_sign(x, num_bits=options["oar"]["omega"])

    # Adjust layer options
    def make_oar_config(use: bool, oar_lambda: float):
        return {
            "use": use,
            "oar_lambda": oar_lambda,
            "omega": options["oar"]["omega"],
        }

    layer_options = {"INPUT": {"ternarize": ternarize_inputs}}

    for name in ["QRNN_0", "QRNN_1"]:
        layer_options[name] = {
            "activation": activation,
            "oar": make_oar_config(oar, options["oar"]["oar_lambda"]),
            "s": s,
            "τ": t * tern_params[name],
        }

    layer_options["DENSE_0"] = {
        "activation": activation,
        "oar": make_oar_config(oar, options["oar"]["oar_lambda"]),
        "s": 1.0,
        "τ": t * tern_params["DENSE_0"],
    }

    layer_options["DENSE_OUT"] = {
        "activation": tf.keras.activations.softmax,
        "oar": make_oar_config(use=True, oar_lambda=0.0),
        "s": 1.0,
        "τ": t * tern_params["DENSE_OUT"],
    }

    print("OPTIONS AND LAYER OPTIONS:")
    print(options)
    print(layer_options)

    return train(
        pretrained_weights=pretrained_weights,
        options=options,
        layer_options=layer_options,
    )


def perform_four_step_quant(options) -> str:
    """Performs the complete four-step quantization procedure.

    Args:
        options: Experiment options dict

    Returns:
        Path to checkpoint folder of final step parameters
    """
    pretrained_weights = None
    for step in range(1, 5):
        pretrained_weights = perform_step_in_four_step_quant(
            step=step, pretrained_weights=pretrained_weights, options=options
        )
        options["epochs"] = 1000
    return pretrained_weights
```

### Step 4: Run first test to verify config structure works

Run: `pytest tests/test_mnist_config.py::test_mnist_options_has_required_keys -v`

Expected: PASS (tests the MNIST_OPTIONS dict)

### Step 5: Commit partial progress

```bash
git add experiments/mnist/config.py tests/test_mnist_config.py
git commit -m "feat(mnist): add experiments/mnist/config.py with options and layer config"
```

---

## Task 4: Create experiments/mnist/training.py

**Files:**
- Create: `experiments/mnist/training.py`
- Test: `tests/test_mnist_training.py`

### Step 1: Write failing test for training module

Create `tests/test_mnist_training.py`:

```python
"""Tests for experiments/mnist/training.py."""

import tensorflow as tf


def test_configure_environment_returns_strategy():
    from experiments.mnist.training import configure_environment

    strategy, dtype = configure_environment()
    assert isinstance(strategy, tf.distribute.Strategy)
    assert dtype == tf.float32


def test_configure_environment_returns_one_device_strategy():
    from experiments.mnist.training import configure_environment

    strategy, _ = configure_environment()
    assert isinstance(strategy, tf.distribute.OneDeviceStrategy)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_mnist_training.py -v`

Expected: FAIL with `ModuleNotFoundError`

### Step 3: Create experiments/mnist/training.py

Create `experiments/mnist/training.py`:

```python
"""MNIST training utilities."""

import os
from datetime import datetime

import tensorflow as tf

from oar import ReservoirHistogramCallback, reset_stat_weights
from experiments.mnist.data import get_datasets
from experiments.mnist.model import get_model

RUNS_DIR = "runs/"
TB_LOGS_DIR = "logs/tensorboard/"
CKPT_DIR = "checkpoints/"
RECORD_CKPTS = True


def configure_environment():
    """Configures the environment by selecting the datatype and device strategy.

    Returns:
        Tuple of (strategy, dtype)
    """
    dtype = tf.float32

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(str(e))

        print(gpus)
        device = gpus[0].name[17:]
        print("Running single gpu: {}".format(device))
        strategy = tf.distribute.OneDeviceStrategy(device=device)
    else:
        device = tf.config.list_physical_devices("CPU")[0].name[17:]
        print("Running on CPU: {}".format(device))
        strategy = tf.distribute.OneDeviceStrategy(device=device)

    return strategy, dtype


def train(pretrained_weights: str | None, options, layer_options) -> str:
    """Runs training for a number of epochs.

    Loads pretrained weights, builds model, and runs training.
    Also runs evaluation over test set at the end.

    Args:
        pretrained_weights: Path to checkpoints folder for initialization
        options: Training options dict
        layer_options: Per-layer options dict

    Returns:
        Path to checkpoints folder of trained model
    """
    strategy, _ = configure_environment()

    now = datetime.now()
    RUN_DIR = (
        RUNS_DIR + now.strftime("%Y%m") + "/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    )

    BATCHSIZE = options["batch_size"]
    ds_train, ds_val, ds_test = get_datasets(
        batch_size=BATCHSIZE, enlarge=options["enlarge"]
    )

    with strategy.scope():
        model = get_model(options, layer_options)

        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)
            print("Restored pretrained weights from {}.".format(pretrained_weights))

        reset_stat_weights(model)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=options["learning_rate"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    # TensorBoard callback
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=(RUN_DIR + TB_LOGS_DIR),
        histogram_freq=1,
        update_freq="epoch",
    )
    reservoir_cb = ReservoirHistogramCallback(log_dir=(RUN_DIR + TB_LOGS_DIR))

    # Learning rate schedule
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        tf.keras.optimizers.schedules.CosineDecay(
            options["learning_rate"], 100, alpha=0.1
        ),
        verbose=0,
    )

    if RECORD_CKPTS:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=(RUN_DIR + CKPT_DIR),
            save_weights_only=True,
            save_best_only=False,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        )
    else:
        ckpt_callback = None

    try:
        model.fit(
            ds_train,
            epochs=options["epochs"],
            validation_data=ds_val,
            callbacks=[tb_callback, reservoir_cb, ckpt_callback, lr_callback],
            verbose=1,
        )
    except Exception as e:
        print(e)

    print("\nRUNNING EVALUATION OVER TEST SET\n")
    try:
        model.evaluate(
            ds_test,
            verbose=1,
        )
    except Exception as e:
        print(e)

    return RUN_DIR + CKPT_DIR
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_mnist_training.py -v`

Expected: All 2 tests PASS

### Step 5: Commit

```bash
git add experiments/mnist/training.py tests/test_mnist_training.py
git commit -m "feat(mnist): add experiments/mnist/training.py with train function"
```

---

## Task 5: Update experiments/mnist/__init__.py with full exports

**Files:**
- Modify: `experiments/mnist/__init__.py`

### Step 1: Update __init__.py with all exports

Update `experiments/mnist/__init__.py`:

```python
"""MNIST RNN experiment with OAR regularization."""

from experiments.mnist.data import get_datasets, normalize_img, resize, augment
from experiments.mnist.model import get_model
from experiments.mnist.config import (
    MNIST_OPTIONS,
    ENLARGED_MNIST_OPTIONS,
    get_default_layer_options,
    get_model_parameter_stats,
    perform_step_in_four_step_quant,
    perform_four_step_quant,
)
from experiments.mnist.training import configure_environment, train

__all__ = [
    # Data
    "get_datasets",
    "normalize_img",
    "resize",
    "augment",
    # Model
    "get_model",
    # Config
    "MNIST_OPTIONS",
    "ENLARGED_MNIST_OPTIONS",
    "get_default_layer_options",
    "get_model_parameter_stats",
    "perform_step_in_four_step_quant",
    "perform_four_step_quant",
    # Training
    "configure_environment",
    "train",
]
```

### Step 2: Run all mnist tests

Run: `pytest tests/test_mnist_*.py -v`

Expected: All tests PASS

### Step 3: Commit

```bash
git add experiments/mnist/__init__.py
git commit -m "feat(mnist): update __init__.py with full experiment exports"
```

---

## Task 6: Remove get_default_layer_options_from_options from oar/

**Files:**
- Modify: `oar/training.py`
- Modify: `oar/__init__.py`
- Modify: `tests/test_phase1_naming.py`
- Modify: `tests/test_oar_serialization.py`

### Step 1: Update oar/training.py to remove the function

Update `oar/training.py` to remove `get_default_layer_options_from_options`:

```python
"""OAR training utilities and model classes."""

import tensorflow as tf
from keras.engine import data_adapter


@tf.keras.utils.register_keras_serializable(package="OAR")
class OARModel(tf.keras.models.Model):
    """
    Keras Model subclass with optional gradient logging.

    Provides a custom train_step that can optionally log gradient
    norms and statistics for debugging training dynamics.

    Args:
        log_gradients: If True, compute and log gradient statistics
    """

    def __init__(self, *args, log_gradients=False, **kwargs):
        super(OARModel, self).__init__(*args, **kwargs)
        self.log_gradients = log_gradients

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)

        self._validate_target_and_loss(y, loss)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        output = self.compute_metrics(x, y, y_pred, sample_weight)

        if self.log_gradients:
            # Gradient logging (commented out by default for performance)
            # Uncomment specific blocks as needed for debugging
            pass

        return output
```

### Step 2: Update oar/__init__.py to remove the export

Update `oar/__init__.py` to remove `get_default_layer_options_from_options`:

```python
"""
Overflow-Aware Activity Regularization (OAR) Package.

A TensorFlow/Keras framework for training quantized neural networks
with overflow-aware regularization for TFHE inference.
"""

from oar.regularizers import OAR1, OAR2, OARRegularizer, oar_penalty_fn, compute_oar_metric
from oar.activations import sign_ste_tanh, mod_sign, TrackedActivation
from oar.quantizers import TernarizationWithThreshold, ternarize_tensor_with_threshold
from oar.callbacks import (
    ReservoirHistogramCallback,
    reset_stat_weights,
    _reservoir_update,
    RESERVOIR_UPDATE_EVERY,
)
from oar.layers import (
    QRNNWithOAR,
    QSimpleRNNCellWithOAR,
    QDenseWithOAR,
    Downsampling,
)
from oar.training import OARModel

__version__ = "0.1.0"

__all__ = [
    # Regularizers
    "OAR1",
    "OAR2",
    "OARRegularizer",
    "oar_penalty_fn",
    "compute_oar_metric",
    # Activations
    "sign_ste_tanh",
    "mod_sign",
    "TrackedActivation",
    # Quantizers
    "TernarizationWithThreshold",
    "ternarize_tensor_with_threshold",
    # Callbacks
    "ReservoirHistogramCallback",
    "reset_stat_weights",
    "_reservoir_update",
    "RESERVOIR_UPDATE_EVERY",
    # Layers
    "QRNNWithOAR",
    "QSimpleRNNCellWithOAR",
    "QDenseWithOAR",
    "Downsampling",
    # Training
    "OARModel",
]
```

### Step 3: Update test_phase1_naming.py

Update the test to import from experiments.mnist:

```python
def test_default_layer_options_use_omega_and_oar_lambda():
    from experiments.mnist.config import get_default_layer_options

    options = {
        "oar": {"omega": 6, "oar_lambda": 1e-4},
        "s": 1.0,
        "t": 1.0,
    }
    layer_options = get_default_layer_options(options)
    assert layer_options["QRNN_0"]["oar"]["omega"] == 6
    assert layer_options["QRNN_0"]["oar"]["oar_lambda"] == 1e-4
```

### Step 4: Update test_oar_serialization.py

Update the test to import from experiments.mnist:

```python
def test_get_default_layer_options_from_oar():
    from experiments.mnist.config import get_default_layer_options

    options = {
        "oar": {"omega": 6, "oar_lambda": 1e-4},
    }
    layer_options = get_default_layer_options(options)
    assert layer_options["QRNN_0"]["oar"]["omega"] == 6
    assert layer_options["QRNN_0"]["oar"]["oar_lambda"] == 1e-4
```

### Step 5: Run tests to verify

Run: `pytest tests/ -v`

Expected: All tests PASS

### Step 6: Commit

```bash
git add oar/training.py oar/__init__.py tests/test_phase1_naming.py tests/test_oar_serialization.py
git commit -m "refactor(oar): remove MNIST-specific get_default_layer_options_from_options"
```

---

## Task 7: Update main.py to use experiments/mnist

**Files:**
- Modify: `main.py`

### Step 1: Rewrite main.py as thin experiment runner

Update `main.py`:

```python
"""Entry point for OAR experiments."""

import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit("autoclustering")

tf.random.set_seed(1997)  # For experimental reproducibility

from experiments.mnist import (
    MNIST_OPTIONS,
    ENLARGED_MNIST_OPTIONS,
    perform_four_step_quant,
    perform_step_in_four_step_quant,
    get_default_layer_options,
    get_model_parameter_stats,
)
from export_mnist_weights_h5 import export_mnist_weights
from export_mnist import extract_ternarized_mnist_test_dataset


def train_quantize_extract_MNIST_RNN() -> str:
    """Runs four-step quantization and exports weights for standard MNIST."""
    import copy
    options = copy.deepcopy(MNIST_OPTIONS)
    final_parameters = perform_four_step_quant(options)
    export_mnist_weights(final_parameters, options)
    return final_parameters


def train_quantize_extract_enlarged_MNIST_RNN() -> str:
    """Runs four-step quantization and exports weights for enlarged MNIST."""
    import copy
    options = copy.deepcopy(ENLARGED_MNIST_OPTIONS)
    final_parameters = perform_four_step_quant(options)
    export_mnist_weights(final_parameters, options)
    return final_parameters


def evaluation_with_and_without_oar2():
    """Evaluates models with and without OAR2 across different bit widths."""
    import copy
    third_step_options = copy.deepcopy(MNIST_OPTIONS)
    third_step = None
    for step in range(1, 4):
        third_step = perform_step_in_four_step_quant(
            step=step, pretrained_weights=third_step, options=third_step_options
        )
        third_step_options["epochs"] = 1000
    for i in range(2):
        for ω in range(3, 9):
            cur_options = copy.deepcopy(third_step_options)
            cur_options["oar"]["omega"] = ω
            cur_options["oar"]["oar_lambda"] = i * 1e-3
            perform_step_in_four_step_quant(
                step=4, pretrained_weights=third_step, options=cur_options
            )


def evaluation_different_oar_regularization_rates():
    """Evaluates different OAR regularization rates."""
    import copy
    third_step_options = copy.deepcopy(MNIST_OPTIONS)
    third_step = None
    for step in range(1, 4):
        third_step = perform_step_in_four_step_quant(
            step=step, pretrained_weights=third_step, options=third_step_options
        )
        third_step_options["epochs"] = 1000
    rates = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    bits = [5, 6]
    for ω in bits:
        for oar_lambda in rates:
            cur_options = copy.deepcopy(third_step_options)
            cur_options["oar"]["omega"] = ω
            cur_options["oar"]["oar_lambda"] = oar_lambda
            perform_step_in_four_step_quant(
                step=4, pretrained_weights=third_step, options=cur_options
            )


if __name__ == "__main__":
    train_quantize_extract_MNIST_RNN()
    # train_quantize_extract_enlarged_MNIST_RNN()
    # evaluation_with_and_without_oar2()
    # evaluation_different_oar_regularization_rates()
    # extract_ternarized_mnist_test_dataset()
    print("End!")
```

### Step 2: Run main.py with minimal training to verify

Run: `python -c "from experiments.mnist import MNIST_OPTIONS; print(MNIST_OPTIONS)"`

Expected: Prints the MNIST_OPTIONS dict without errors

### Step 3: Commit

```bash
git add main.py
git commit -m "refactor(main): use experiments.mnist module for MNIST training"
```

---

## Task 8: Update export_mnist_weights_h5.py

**Files:**
- Modify: `export_mnist_weights_h5.py`

### Step 1: Update imports

Update `export_mnist_weights_h5.py`:

```python
import os
from oar import reset_stat_weights
from experiments.mnist import get_model, get_default_layer_options


def export_mnist_weights(pretrained_weights: str, options):
    """Given the pretrained weights folder, extract the weights in hdf5 format.

    Args:
        pretrained_weights: Path to checkpoints folder containing model parameters
        options: Options for model
    """
    model = get_model(options, get_default_layer_options(options))

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
        print("Restored pretrained weights from {}.".format(pretrained_weights))

    # Reset the stat variables
    reset_stat_weights(model)

    h5_dir = pretrained_weights + "hdf5/"
    if not os.path.exists(h5_dir):
        os.makedirs(h5_dir)
    h5_filepath = h5_dir + "weights.hdf5"

    print("Saving weights from -> " + pretrained_weights)
    print("Saving weights to   -> " + h5_filepath)
    model.save_weights(h5_filepath, overwrite=False, save_format="h5")
    print("Completed. Goodbye.")
```

### Step 2: Verify import works

Run: `python -c "from export_mnist_weights_h5 import export_mnist_weights; print('OK')"`

Expected: Prints "OK"

### Step 3: Commit

```bash
git add export_mnist_weights_h5.py
git commit -m "refactor(export): use experiments.mnist imports"
```

---

## Task 9: Delete mnist_rnn_model.py

**Files:**
- Delete: `mnist_rnn_model.py`

### Step 1: Verify no direct imports remain

Run: `grep -r "from mnist_rnn_model" . --include="*.py" | grep -v experiments/`

Expected: No output (no remaining imports)

### Step 2: Delete the file

Run: `rm mnist_rnn_model.py`

### Step 3: Commit

```bash
git add -A
git commit -m "chore: remove deprecated mnist_rnn_model.py (moved to experiments/mnist/model.py)"
```

---

## Task 10: Run full test suite and integration test

**Files:** None (validation only)

### Step 1: Run all tests

Run: `pytest tests/ -v`

Expected: All tests PASS

### Step 2: Run a quick training smoke test (1 epoch)

Run:
```python
python -c "
import copy
from experiments.mnist import MNIST_OPTIONS, perform_step_in_four_step_quant
options = copy.deepcopy(MNIST_OPTIONS)
options['epochs'] = 1
perform_step_in_four_step_quant(step=1, pretrained_weights=None, options=options)
print('Smoke test passed!')
"
```

Expected: Model trains for 1 epoch, prints "Smoke test passed!"

### Step 3: Commit final verification

```bash
git add -A
git commit -m "test: verify Phase 3 refactoring - all tests pass"
```

---

## Summary

**Files Created:**
- `experiments/__init__.py`
- `experiments/mnist/__init__.py`
- `experiments/mnist/data.py`
- `experiments/mnist/model.py`
- `experiments/mnist/config.py`
- `experiments/mnist/training.py`
- `tests/test_mnist_data.py`
- `tests/test_mnist_model.py`
- `tests/test_mnist_config.py`
- `tests/test_mnist_training.py`

**Files Modified:**
- `main.py` - simplified to thin experiment runner
- `export_mnist_weights_h5.py` - updated imports
- `oar/training.py` - removed MNIST-specific function
- `oar/__init__.py` - removed MNIST-specific export
- `tests/test_phase1_naming.py` - updated imports
- `tests/test_oar_serialization.py` - updated imports

**Files Deleted:**
- `mnist_rnn_model.py`

**Key Changes from Original Plan:**
1. Added `experiments/mnist/training.py` to hold `configure_environment()` and `train()` - these were in `main.py` but are MNIST-specific
2. Removed `get_default_layer_options_from_options` from `oar/training.py` - it was MNIST-specific and doesn't belong in the core framework
3. Added `ENLARGED_MNIST_OPTIONS` constant for 128x128 experiments
4. Renamed `get_default_layer_options_from_options` to `get_default_layer_options` (cleaner)

**Code Simplifications Applied:**
1. `get_default_layer_options()` - uses loop over layer names instead of repeating config 4 times
2. `perform_step_in_four_step_quant()` - layer_options construction uses same loop pattern
3. `get_datasets()` - extracted `_prepare_dataset()` helper to avoid repeating train/val/test pipeline
4. Test files use pytest fixtures (`default_options`, `default_layer_options`) to reduce boilerplate
5. Removed useless `test_train_function_exists` (only tested `callable(train)`)
6. Added edge case tests: `test_augment_returns_same_shape`, `test_normalize_img_preserves_zero`, `test_get_model_has_correct_input_shape_128x128`

---

Plan complete and saved to `docs/plans/2026-02-08-phase3-experiments-mnist.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
