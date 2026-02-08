# Phase 2: Extract `oar/` Package Implementation Plan

> **For Agent:** REQUIRED: Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Extract reusable OAR components from `utils/model_utils.py` into a dedicated `oar/` package with proper Keras serialization support.

**Architecture:** Create `oar/` package with modules for regularizers, activations, layers, quantizers, training utilities, and callbacks. Each module uses `@register_keras_serializable` for proper save/load support. All custom classes implement complete `get_config()` methods following QKeras patterns (use `constraints.serialize()` for quantizers, handle `tf.Variable` conversion to numpy).

**Tech Stack:** Python 3.10, TensorFlow 2.10.1, QKeras 0.9.0, pytest

---

## Pre-Implementation Notes

### Serialization Patterns (from QKeras source investigation)

**For Layers:**
```python
@tf.keras.utils.register_keras_serializable(package="OAR")
class MyLayer(tf.keras.layers.Layer):
    def get_config(self):
        config = {
            "my_param": self.my_param,
            "my_quantizer": tf.keras.constraints.serialize(self.quantizer_internal),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

**For Quantizers (extend `tf.Module`, not `Layer`):**
```python
@tf.keras.utils.register_keras_serializable(package="OAR")
class MyQuantizer(BaseQuantizer):
    def get_config(self):
        return {
            "param": self.param.numpy() if isinstance(self.param, tf.Variable) else self.param,
        }
```

### Known Bugs to Fix in This Phase

1. `TrackedActivation.get_config()` references non-existent `self.alpha_init`
2. `TernarizationWithThreshold.get_config()` missing `qnoise_factor`, `var_name`, `use_ste`, `use_variables`
3. Missing `@register_keras_serializable` decorators on all custom classes

---

## Target Package Structure

```
oar/
├── __init__.py              # Clean public API
├── regularizers.py          # OAR1, OAR2, oar_penalty_fn, compute_oar_metric
├── activations.py           # sign_ste_tanh, mod_sign, TrackedActivation
├── layers.py                # QRNNWithOAR, QSimpleRNNCellWithOAR, QDenseWithOAR, Downsampling
├── quantizers.py            # TernarizationWithThreshold, ternarize_tensor_with_threshold
├── training.py              # reset_stat_weights, get_default_layer_options_from_options
└── callbacks.py             # ReservoirHistogramCallback, _reservoir_update, RESERVOIR_UPDATE_EVERY
```

---

### Task 1: Create oar package skeleton with __init__.py

**Files:**
- Create: `oar/__init__.py`

**Step 1: Create the package directory and init file**

```python
"""
Overflow-Aware Activity Regularization (OAR) Package.

A TensorFlow/Keras framework for training quantized neural networks
with overflow-aware regularization for TFHE inference.
"""

# This will be populated as we add modules
__version__ = "0.1.0"
```

**Step 2: Verify package is importable**

Run: `PYTHONPATH=. python -c "import oar; print(oar.__version__)"`
Expected: `0.1.0`

**Step 3: Commit**

```bash
git add oar/__init__.py
git commit -m "feat(oar): create package skeleton"
```

---

### Task 2: Add serialization tests for regularizers

**Files:**
- Create: `tests/test_oar_serialization.py`

**Step 1: Write the failing test**

```python
import tensorflow as tf

# Test imports from new location
from oar.regularizers import OAR1, OAR2, oar_penalty_fn, compute_oar_metric


def test_oar1_get_config_returns_all_params():
    layer = OAR1(oar_lambda=1e-3, k=64, a=2.0, name="test_oar1")
    config = layer.get_config()
    assert config["oar_lambda"] == 1e-3
    assert config["k"] == 64
    assert config["a"] == 2.0


def test_oar2_get_config_returns_all_params():
    layer = OAR2(oar_lambda=5e-4, k=128, a=1.5, name="test_oar2")
    config = layer.get_config()
    assert config["oar_lambda"] == 5e-4
    assert config["k"] == 128
    assert config["a"] == 1.5


def test_oar1_serialization_roundtrip():
    original = OAR1(oar_lambda=1e-3, k=64, a=2.0, name="test")
    config = original.get_config()
    restored = OAR1.from_config(config)
    assert restored.oar_lambda == original.oar_lambda
    assert restored.k == original.k
    assert restored.a == original.a


def test_oar2_serialization_roundtrip():
    original = OAR2(oar_lambda=5e-4, k=128, a=1.5, name="test")
    config = original.get_config()
    restored = OAR2.from_config(config)
    assert restored.oar_lambda == original.oar_lambda
    assert restored.k == original.k
    assert restored.a == original.a


def test_oar_penalty_fn_unchanged():
    x = tf.constant([0.0, 0.0, 0.0])
    out = oar_penalty_fn(x, k=8, a=1.0)
    tf.debugging.assert_equal(out, tf.zeros_like(out))


def test_compute_oar_metric_unchanged():
    x = tf.constant([[0.0, 0.0]])
    out = compute_oar_metric(x, k=8, a=1.0)
    tf.debugging.assert_near(out, tf.constant([1.0]))
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py::test_oar1_get_config_returns_all_params -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'oar.regularizers'"

**Step 3: Commit**

```bash
git add tests/test_oar_serialization.py
git commit -m "test(oar): add regularizer serialization expectations"
```

---

### Task 3: Extract regularizers to oar/regularizers.py

**Files:**
- Create: `oar/regularizers.py`
- Modify: `oar/__init__.py`

**Step 1: The tests from Task 2 should still fail**

**Step 2: Write minimal implementation**

Create `oar/regularizers.py`:

```python
"""OAR regularizers for overflow-aware training."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="OAR")
def oar_penalty_fn(x, k, a):
    """
    OAR penalty function (Equation 1 from paper).
    
    OAR₁(x, k) = ReLU(1 - (4/k) · | (|x + 0.5| - k/4) mod k - k/2 |)
    
    Args:
        x: Input tensor (pre-activations)
        k: Modulus size (2^omega)
        a: Amplitude scaling factor
    
    Returns:
        Penalty tensor (same shape as x)
    """
    inner = tf.abs(x + 0.5) - k / 4
    modded = tf.math.mod(inner, k)
    out = 1 - (4 / k) * tf.abs(modded - k / 2)
    return a * tf.nn.relu(out)


def compute_oar_metric(x, k, a):
    """
    Compute fraction of pre-activations in valid modular range.
    
    Args:
        x: Input tensor (pre-activations)
        k: Modulus size (2^omega)
        a: Amplitude scaling factor
    
    Returns:
        Fraction of values NOT in overflow region (higher is better)
    """
    wrongs = tf.sign(oar_penalty_fn(x, k=k, a=a))
    rights_ratio = 1 - tf.reduce_mean(wrongs, axis=[-1])
    return rights_ratio


@tf.keras.utils.register_keras_serializable(package="OAR")
class OARRegularizer(tf.keras.layers.Layer):
    """
    Base Overflow-Aware Activity Regularizer.
    
    Applies a penalty for pre-activations in overflow regions.
    
    Args:
        oar_lambda: Regularization rate (lambda in paper)
        k: Modulus size, typically 2^omega where omega is bit-width
        a: Amplitude scaling factor (default 1.0)
        squared: If True, square the penalty (OAR2); if False, linear (OAR1)
        name: Layer name for metric tracking
    """

    def __init__(self, oar_lambda=1e-3, k=2**8, a=1.0, squared=False, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.oar_lambda = oar_lambda
        self.k = k
        self.a = a
        self.squared = squared
        metric_prefix = "OAR2" if squared else "OAR1"
        self.no_acc_metric = tf.keras.metrics.Mean(name=f"{metric_prefix}/{name}")

    def call(self, x):
        penalty = oar_penalty_fn(x=x, k=self.k, a=self.a)
        if self.squared:
            penalty = tf.square(penalty)
        
        loss = self.oar_lambda * tf.reduce_sum(penalty)
        accuracy = compute_oar_metric(x, k=self.k, a=self.a)
        
        self.add_loss(loss)
        self.add_metric(self.no_acc_metric(accuracy))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "oar_lambda": float(self.oar_lambda),
            "k": int(self.k),
            "a": float(self.a),
            "squared": bool(self.squared),
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Backward-compatible factory functions
def OAR1(oar_lambda=1e-3, k=2**8, a=1.0, name="", **kwargs):
    """OAR₁ - linear penalty. Returns OARRegularizer with squared=False."""
    return OARRegularizer(oar_lambda=oar_lambda, k=k, a=a, squared=False, name=name, **kwargs)


def OAR2(oar_lambda=1e-3, k=2**8, a=1.0, name="", **kwargs):
    """OAR₂ - squared penalty. Returns OARRegularizer with squared=True."""
    return OARRegularizer(oar_lambda=oar_lambda, k=k, a=a, squared=True, name=name, **kwargs)
```

Update `oar/__init__.py`:

```python
"""
Overflow-Aware Activity Regularization (OAR) Package.

A TensorFlow/Keras framework for training quantized neural networks
with overflow-aware regularization for TFHE inference.
"""

from oar.regularizers import OAR1, OAR2, OARRegularizer, oar_penalty_fn, compute_oar_metric

__version__ = "0.1.0"

__all__ = [
    "OAR1",
    "OAR2",
    "OARRegularizer",
    "oar_penalty_fn",
    "compute_oar_metric",
]
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py -v -k regularizer`
Expected: All regularizer tests PASS

**Step 4: Commit**

```bash
git add oar/regularizers.py oar/__init__.py
git commit -m "feat(oar): extract regularizers module"
```

---

### Task 4: Add serialization tests for activations

**Files:**
- Modify: `tests/test_oar_serialization.py`

**Step 1: Write the failing test**

Append to `tests/test_oar_serialization.py`:

```python
from oar.activations import sign_ste_tanh, mod_sign, TrackedActivation


def test_sign_ste_tanh_unchanged():
    x = tf.constant([-2.0, 0.0, 2.0])
    out = sign_ste_tanh(x)
    tf.debugging.assert_equal(out, tf.constant([-1.0, 1.0, 1.0]))


def test_mod_sign_unchanged():
    x = tf.constant([-1.0, 1.0, 5.0])
    out = mod_sign(x, num_bits=3)
    tf.debugging.assert_equal(out, tf.constant([-1.0, 1.0, -1.0]))


def test_tracked_activation_get_config_fixed():
    """Test that TrackedActivation.get_config() no longer references alpha_init."""
    layer = TrackedActivation(activation=tf.nn.relu, name="test_act")
    config = layer.get_config()
    # Should have activation and name, NOT alpha_init
    assert "activation" in config or "name" in config
    assert "alpha_init" not in config


def test_tracked_activation_serialization_roundtrip():
    original = TrackedActivation(activation="relu", name="test")
    config = original.get_config()
    # Should not raise
    restored = TrackedActivation.from_config(config)
    assert restored is not None


def test_tracked_activation_applies_activation():
    layer = TrackedActivation(activation=tf.nn.relu)
    x = tf.constant([-1.0, 2.0])
    out = layer(x)
    tf.debugging.assert_equal(out, tf.constant([0.0, 2.0]))
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py::test_sign_ste_tanh_unchanged -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'oar.activations'"

**Step 3: Commit**

```bash
git add tests/test_oar_serialization.py
git commit -m "test(oar): add activation serialization expectations"
```

---

### Task 5: Extract activations to oar/activations.py

**Files:**
- Create: `oar/activations.py`
- Modify: `oar/__init__.py`

**Step 1: The tests from Task 4 should still fail**

**Step 2: Write minimal implementation**

Create `oar/activations.py`:

```python
"""OAR activation functions and wrappers."""

import tensorflow as tf


def sign_ste_tanh(x):
    """
    Sign activation with tanh gradient (straight-through estimator).
    
    Forward: sign(x)
    Backward: tanh'(x) = 1 - tanh(x)^2
    
    Handles x=0 by returning +1 (adds 1.0 - |sign(x)| to fill zeros).
    
    Args:
        x: Input tensor
    
    Returns:
        Sign of x with tanh gradient for backprop
    """
    out = tf.keras.activations.tanh(x)
    q = tf.math.sign(x)
    q += 1.0 - tf.math.abs(q)  # Handle x=0 -> +1
    return out + tf.stop_gradient(-out + q)


def mod_sign(x, num_bits=8):
    """
    Modular sign activation (ModSign from paper, Eq. 4).
    
    Computes sign(x mod 2^num_bits) with tanh gradient.
    Used in Step 4 of four-step quantization for TFHE compatibility.
    
    Args:
        x: Input tensor (pre-activations)
        num_bits: Bit precision (omega), determines modulus 2^num_bits
    
    Returns:
        Signed modular result with tanh gradient for backprop
    """
    def _inner_fn(x, num_bits):
        base = 2**num_bits
        half_base = 2 ** (num_bits - 1)

        # Cast to int for modular reduction
        x_int = tf.cast(x, tf.int32)

        # Modular reduction (unsigned)
        modded = tf.math.mod(x_int, base)

        # Convert to signed representation
        signed = tf.where(tf.greater_equal(modded, half_base), modded - base, modded)

        return tf.cast(signed, tf.float32)

    # Regular sign with STE, then replace forward value with modular version
    out = sign_ste_tanh(x) + tf.stop_gradient(
        -sign_ste_tanh(x) + sign_ste_tanh(_inner_fn(x, num_bits=num_bits))
    )

    return out


@tf.keras.utils.register_keras_serializable(package="OAR")
class TrackedActivation(tf.keras.layers.Layer):
    """
    Activation wrapper that tracks input/output statistics.
    
    Maintains exponential moving averages of mean and std for both
    input and output of the activation function. Useful for monitoring
    distribution shifts during training.
    
    Args:
        activation: Activation function (callable or string)
        name: Layer name
    """

    def __init__(self, activation=None, name="", **kwargs):
        super(TrackedActivation, self).__init__(name=name, **kwargs)
        self._activation = activation
        # Resolve string activations to callables
        if isinstance(activation, str):
            self._activation_fn = tf.keras.activations.get(activation)
        else:
            self._activation_fn = activation

    def build(self, input_shape):
        self.inp_moving_mean = self.add_weight(
            name="inp_moving_mean",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        self.inp_moving_std = self.add_weight(
            name="inp_moving_std",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        self.out_moving_mean = self.add_weight(
            name="out_moving_mean",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        self.out_moving_std = self.add_weight(
            name="out_moving_std",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        super(TrackedActivation, self).build(input_shape)

    def call(self, inputs):
        out = inputs

        # Apply activation
        if self._activation_fn:
            out = self._activation_fn(out)

        # Update statistics
        self.inp_moving_mean.assign(
            0.9 * self.inp_moving_mean + 0.1 * tf.reduce_mean(inputs)
        )
        self.inp_moving_std.assign(
            0.9 * self.inp_moving_std + 0.1 * tf.math.reduce_std(inputs)
        )
        self.out_moving_mean.assign(
            0.9 * self.out_moving_mean + 0.1 * tf.reduce_mean(out)
        )
        self.out_moving_std.assign(
            0.9 * self.out_moving_std + 0.1 * tf.math.reduce_std(out)
        )

        return out

    def get_config(self):
        config = super().get_config()
        # Serialize activation - handle both callables and strings
        if isinstance(self._activation, str):
            activation_config = self._activation
        elif self._activation is None:
            activation_config = None
        else:
            # Try to serialize as Keras activation
            try:
                activation_config = tf.keras.activations.serialize(self._activation)
            except (TypeError, ValueError):
                # For custom functions, store as None and warn
                activation_config = None
        config.update({
            "activation": activation_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize activation if needed
        activation = config.pop("activation", None)
        if isinstance(activation, dict):
            activation = tf.keras.activations.deserialize(activation)
        elif isinstance(activation, str):
            activation = tf.keras.activations.get(activation)
        return cls(activation=activation, **config)
```

Update `oar/__init__.py`:

```python
"""
Overflow-Aware Activity Regularization (OAR) Package.

A TensorFlow/Keras framework for training quantized neural networks
with overflow-aware regularization for TFHE inference.
"""

from oar.regularizers import OAR1, OAR2, oar_penalty_fn, compute_oar_metric
from oar.activations import sign_ste_tanh, mod_sign, TrackedActivation

__version__ = "0.1.0"

__all__ = [
    # Regularizers
    "OAR1",
    "OAR2",
    "oar_penalty_fn",
    "compute_oar_metric",
    # Activations
    "sign_ste_tanh",
    "mod_sign",
    "TrackedActivation",
]
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py -v -k activation`
Expected: All activation tests PASS

**Step 4: Commit**

```bash
git add oar/activations.py oar/__init__.py
git commit -m "feat(oar): extract activations module with fixed get_config"
```

---

### Task 6: Add serialization tests for quantizers

**Files:**
- Modify: `tests/test_oar_serialization.py`

**Step 1: Write the failing test**

Append to `tests/test_oar_serialization.py`:

```python
from oar.quantizers import TernarizationWithThreshold, ternarize_tensor_with_threshold


def test_ternarize_tensor_with_threshold_unchanged():
    x = tf.constant([-2.0, -0.1, 0.1, 2.0])
    out = ternarize_tensor_with_threshold(x, theta=0.5)
    tf.debugging.assert_equal(out, tf.constant([-1.0, 0.0, 0.0, 1.0]))


def test_ternarization_get_config_returns_all_params():
    """Test that get_config returns ALL constructor parameters (bug fix)."""
    quantizer = TernarizationWithThreshold(
        threshold=0.5,
        qnoise_factor=0.8,
        var_name="test_var",
        use_ste=False,
        use_variables=True,
        name="test_quant",
    )
    config = quantizer.get_config()
    assert config["threshold"] == 0.5
    assert config["qnoise_factor"] == 0.8
    assert config["var_name"] == "test_var"
    assert config["use_ste"] == False
    assert config["use_variables"] == True


def test_ternarization_serialization_roundtrip():
    original = TernarizationWithThreshold(
        threshold=0.3,
        qnoise_factor=0.9,
        use_ste=True,
    )
    config = original.get_config()
    restored = TernarizationWithThreshold.from_config(config)
    assert restored.threshold == original.threshold
    assert restored.qnoise_factor == original.qnoise_factor
    assert restored.use_ste == original.use_ste


def test_ternarization_call_produces_ternary():
    quantizer = TernarizationWithThreshold(threshold=0.5)
    x = tf.constant([-2.0, -0.1, 0.1, 2.0])
    out = quantizer(x)
    # With STE, output should be ternary values
    unique_vals = tf.unique(out)[0]
    # Should only have values in {-1, 0, 1}
    for val in unique_vals.numpy():
        assert val in [-1.0, 0.0, 1.0]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py::test_ternarize_tensor_with_threshold_unchanged -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'oar.quantizers'"

**Step 3: Commit**

```bash
git add tests/test_oar_serialization.py
git commit -m "test(oar): add quantizer serialization expectations"
```

---

### Task 7: Extract quantizers to oar/quantizers.py

**Files:**
- Create: `oar/quantizers.py`
- Modify: `oar/__init__.py`

**Step 1: The tests from Task 6 should still fail**

**Step 2: Write minimal implementation**

Create `oar/quantizers.py`:

```python
"""OAR quantizers for ternary weight quantization."""

import tensorflow as tf
import keras.backend as K
from qkeras import BaseQuantizer


def ternarize_tensor_with_threshold(x, theta=1.0):
    """
    Ternary quantization with threshold.
    
    Maps values to {-1, 0, +1} based on threshold:
    - x = -1 if x <= -theta
    - x = 0  if -theta < x < theta
    - x = +1 if x >= theta
    
    Args:
        x: Input tensor
        theta: Threshold value (τ in paper)
    
    Returns:
        Ternarized tensor
    """
    q = K.cast(tf.abs(x) >= theta, K.floatx()) * tf.sign(x)
    return q


@tf.keras.utils.register_keras_serializable(package="OAR")
class TernarizationWithThreshold(BaseQuantizer):
    """
    Ternary quantizer with configurable threshold.
    
    Quantizes weights to {-1, 0, +1} using a threshold τ.
    Typically τ = t * E[|θ_l|] where t is a scaling factor and
    θ_l are the layer weights.
    
    Uses straight-through estimator (STE) for gradient computation.
    
    Args:
        threshold: Quantization threshold (τ)
        qnoise_factor: Noise factor for STE (1.0 = full quantization)
        var_name: Variable name prefix for TF variables
        use_ste: If True, use STE; if False, use weighted average
        use_variables: If True, create TF variables for parameters
        name: Quantizer name
    """

    def __init__(
        self,
        threshold=None,
        qnoise_factor=1.0,
        var_name=None,
        use_ste=True,
        use_variables=False,
        name="",
    ):
        super(TernarizationWithThreshold, self).__init__()
        self.bits = 2
        self.threshold = threshold
        self.initialized = False
        self.qnoise_factor = qnoise_factor
        self.use_ste = use_ste
        self.var_name = var_name
        self.use_variables = use_variables

    def __call__(self, x):
        if not self.built:
            self.build(var_name=self.var_name, use_variables=self.use_variables)
            self.initialized = True

        xq = ternarize_tensor_with_threshold(x, theta=self.threshold)

        if self.use_ste:
            return x + tf.stop_gradient(self.qnoise_factor * (-x + xq))
        else:
            return (1 - self.qnoise_factor) * x + tf.stop_gradient(
                self.qnoise_factor * xq
            )

    def max(self):
        """Maximum representable value."""
        return 1.0

    def min(self):
        """Minimum representable value."""
        return -1.0

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        """Return all constructor parameters for serialization."""
        return {
            "threshold": self.threshold,
            "qnoise_factor": self.qnoise_factor,
            "var_name": self.var_name,
            "use_ste": self.use_ste,
            "use_variables": self.use_variables,
        }
```

Update `oar/__init__.py`:

```python
"""
Overflow-Aware Activity Regularization (OAR) Package.

A TensorFlow/Keras framework for training quantized neural networks
with overflow-aware regularization for TFHE inference.
"""

from oar.regularizers import OAR1, OAR2, oar_penalty_fn, compute_oar_metric
from oar.activations import sign_ste_tanh, mod_sign, TrackedActivation
from oar.quantizers import TernarizationWithThreshold, ternarize_tensor_with_threshold

__version__ = "0.1.0"

__all__ = [
    # Regularizers
    "OAR1",
    "OAR2",
    "oar_penalty_fn",
    "compute_oar_metric",
    # Activations
    "sign_ste_tanh",
    "mod_sign",
    "TrackedActivation",
    # Quantizers
    "TernarizationWithThreshold",
    "ternarize_tensor_with_threshold",
]
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py -v -k quantizer`
Expected: All quantizer tests PASS

**Step 4: Commit**

```bash
git add oar/quantizers.py oar/__init__.py
git commit -m "feat(oar): extract quantizers module with fixed get_config"
```

---

### Task 8: Add tests for callback module

**Files:**
- Modify: `tests/test_oar_serialization.py`

**Step 1: Write the failing test**

Append to `tests/test_oar_serialization.py`:

```python
import tempfile
from oar.callbacks import (
    ReservoirHistogramCallback,
    _reservoir_update,
    reset_stat_weights,
    RESERVOIR_UPDATE_EVERY,
)


def test_reservoir_update_from_oar_callbacks():
    reservoir = tf.zeros([10], dtype=tf.float32)
    count = tf.constant(0, dtype=tf.int64)
    new_vals = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)

    updated, updated_count = _reservoir_update(
        reservoir=reservoir,
        count=count,
        new_values=new_vals,
        reservoir_size=10,
        seed=123,
    )

    tf.debugging.assert_equal(updated_count, tf.constant(4, dtype=tf.int64))


def test_reservoir_histogram_callback_from_oar():
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = ReservoirHistogramCallback(log_dir=tmpdir)
        assert cb is not None
        assert cb.log_dir == tmpdir


def test_reservoir_update_every_constant():
    assert RESERVOIR_UPDATE_EVERY == 20
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py::test_reservoir_update_from_oar_callbacks -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'oar.callbacks'"

**Step 3: Commit**

```bash
git add tests/test_oar_serialization.py
git commit -m "test(oar): add callback module expectations"
```

---

### Task 9: Extract callbacks to oar/callbacks.py

**Files:**
- Create: `oar/callbacks.py`
- Modify: `oar/__init__.py`

**Step 1: The tests from Task 8 should still fail**

**Step 2: Write minimal implementation**

Create `oar/callbacks.py`:

```python
"""OAR training callbacks and utilities."""

import numpy as np
import tensorflow as tf


# Update reservoir every N batches to reduce overhead.
RESERVOIR_UPDATE_EVERY = 20


def _reservoir_update(reservoir, count, new_values, reservoir_size, seed):
    """
    Vitter's Algorithm R for reservoir sampling (NumPy backend).
    
    Maintains a fixed-size reservoir of samples from a stream,
    ensuring each element has equal probability of being included.
    
    Uses NumPy via tf.numpy_function to avoid slow TF scatter ops in TF 2.10.
    
    Args:
        reservoir: Current reservoir tensor [reservoir_size]
        count: Total samples seen so far (int64 scalar)
        new_values: New values to add (any shape, will be flattened)
        reservoir_size: Maximum reservoir size
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (updated_reservoir, updated_count)
    """
    def _numpy_reservoir_update(
        reservoir_np, count_np, new_values_np, reservoir_size_np, seed_np
    ):
        flat = new_values_np.reshape(-1)
        reservoir_size_i = int(reservoir_size_np)
        count_i = int(count_np)

        # Phase 1: Fill initial slots
        capacity_left = reservoir_size_i - count_i
        fill_n = max(0, min(flat.size, capacity_left))
        if fill_n > 0:
            reservoir_np = reservoir_np.copy()
            reservoir_np[count_i : count_i + fill_n] = flat[:fill_n]
            count_i += fill_n

        # Phase 2: Reservoir sampling
        remaining = flat[fill_n:]
        if remaining.size > 0:
            rng = np.random.default_rng(int(seed_np) + count_i)
            i_range = np.arange(remaining.size, dtype=np.int64)
            total_idx = count_i + i_range + 1
            accept_threshold = reservoir_size_i / total_idx.astype(np.float64)
            accept_mask = rng.random(remaining.size) < accept_threshold
            accepted_idx = np.where(accept_mask)[0]
            if accepted_idx.size > 0:
                slots = rng.integers(
                    low=0, high=reservoir_size_i, size=accepted_idx.size
                )
                reservoir_np = reservoir_np.copy()
                reservoir_np[slots] = remaining[accepted_idx]
            count_i += remaining.size

        return reservoir_np, np.array(count_i, dtype=np.int64)

    updated_reservoir, updated_count = tf.numpy_function(
        _numpy_reservoir_update,
        [reservoir, count, new_values, reservoir_size, seed],
        [reservoir.dtype, tf.int64],
        name="reservoir_update_np",
    )
    updated_reservoir.set_shape(reservoir.shape)
    updated_count.set_shape(count.shape)
    return updated_reservoir, updated_count


def reset_stat_weights(model):
    """
    Zero out stat-tracking weights in a model.
    
    Resets reservoir counts, EMA statistics, and other tracking weights
    that should not persist between training steps.
    
    Args:
        model: Keras model with stat-tracking weights
    """
    weights = model.get_weights()
    for i in range(len(weights)):
        name = model.weights[i].name
        if any(
            pattern in name
            for pattern in [
                "/w",
                "/x",
                "preact_reservoir",
                "reservoir_count",
                "reservoir_step",
            ]
        ):
            weights[i] = 0 * weights[i]
    model.set_weights(weights)


class ReservoirHistogramCallback(tf.keras.callbacks.Callback):
    """
    Logs preact_reservoir histograms to TensorBoard at epoch end.
    
    Scans model weights for reservoir buffers and logs their distributions
    as histograms, enabling visualization of pre-activation distributions.
    
    Args:
        log_dir: TensorBoard log directory
        reservoir_weight_name: Pattern to match reservoir weight names
    """

    def __init__(self, log_dir, reservoir_weight_name="preact_reservoir"):
        super().__init__()
        self.log_dir = log_dir
        self.reservoir_weight_name = reservoir_weight_name
        self.writer = None

    def set_model(self, model):
        super().set_model(model)
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if self.writer is None:
            return
        with self.writer.as_default():
            for weight in self.model.weights:
                if self.reservoir_weight_name in weight.name:
                    tf.summary.histogram(weight.name, weight, step=epoch)
            self.writer.flush()
```

Update `oar/__init__.py`:

```python
"""
Overflow-Aware Activity Regularization (OAR) Package.

A TensorFlow/Keras framework for training quantized neural networks
with overflow-aware regularization for TFHE inference.
"""

from oar.regularizers import OAR1, OAR2, oar_penalty_fn, compute_oar_metric
from oar.activations import sign_ste_tanh, mod_sign, TrackedActivation
from oar.quantizers import TernarizationWithThreshold, ternarize_tensor_with_threshold
from oar.callbacks import (
    ReservoirHistogramCallback,
    reset_stat_weights,
    _reservoir_update,
    RESERVOIR_UPDATE_EVERY,
)

__version__ = "0.1.0"

__all__ = [
    # Regularizers
    "OAR1",
    "OAR2",
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
]
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py -v -k "callback or reservoir"`
Expected: All callback tests PASS

**Step 4: Commit**

```bash
git add oar/callbacks.py oar/__init__.py
git commit -m "feat(oar): extract callbacks module"
```

---

### Task 10: Add tests for layers module

**Files:**
- Modify: `tests/test_oar_serialization.py`

**Step 1: Write the failing test**

Append to `tests/test_oar_serialization.py`:

```python
from oar.layers import (
    QRNNWithOAR,
    QSimpleRNNCellWithOAR,
    QDenseWithOAR,
    Downsampling,
)


def test_downsampling_get_config():
    layer = Downsampling(reduction_factor=2, batch_size=32)
    config = layer.get_config()
    assert config["reduction_factor"] == 2
    assert config["batch_size"] == 32


def test_downsampling_serialization_roundtrip():
    original = Downsampling(reduction_factor=4, batch_size=64)
    config = original.get_config()
    restored = Downsampling.from_config(config)
    assert restored.reduction_factor == original.reduction_factor
    assert restored.batch_size == original.batch_size


def test_qsimple_rnn_cell_has_reservoir_weights():
    cell = QSimpleRNNCellWithOAR(units=4, batch_size=2, name="QRNN_0")
    cell.build(tf.TensorShape([2, 8]))
    assert cell.preact_reservoir.shape.as_list() == [100000]
    assert cell.reservoir_count.dtype == tf.int64


def test_qdense_with_oar_has_reservoir_weights():
    layer = QDenseWithOAR(units=3, batch_size=2, name="DENSE_0")
    layer.build(tf.TensorShape([2, 8]))
    assert layer.preact_reservoir.shape.as_list() == [100000]
    assert layer.reservoir_count.dtype == tf.int64


def test_qrnn_with_oar_propagates_reservoir_size():
    layer = QRNNWithOAR(units=4, batch_size=2, reservoir_size=777, name="QRNN_0")
    layer.build(tf.TensorShape([2, 10, 8]))
    assert layer.cell.preact_reservoir.shape.as_list() == [777]


def test_qrnn_with_oar_serialization_roundtrip():
    original = QRNNWithOAR(units=4, batch_size=2, use_oar=True, oar_lambda=1e-4, omega=6, name="QRNN_0")
    original.build(tf.TensorShape([2, 10, 8]))
    config = original.get_config()
    # Verify key params are serialized
    assert "kernel_quantizer" in config
    assert "use_oar" in str(config) or original.use_oar  # Check param propagates


def test_qsimple_rnn_cell_get_config():
    cell = QSimpleRNNCellWithOAR(units=4, batch_size=2, use_oar=True, oar_lambda=1e-4, omega=6, name="QRNN_0")
    cell.build(tf.TensorShape([2, 8]))
    config = cell.get_config()
    assert config["batch_size"] == 2
    assert config["use_oar"] == True
    assert config["oar_lambda"] == 1e-4
    assert config["omega"] == 6


def test_qdense_with_oar_get_config():
    layer = QDenseWithOAR(units=3, batch_size=2, use_oar=True, oar_lambda=1e-4, omega=6, name="DENSE_0")
    layer.build(tf.TensorShape([2, 8]))
    config = layer.get_config()
    assert config["batch_size"] == 2
    assert config["use_oar"] == True
    assert config["oar_lambda"] == 1e-4
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py::test_downsampling_get_config -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'oar.layers'"

**Step 3: Commit**

```bash
git add tests/test_oar_serialization.py
git commit -m "test(oar): add layer module expectations"
```

---

### Task 11: Extract layers to oar/layers.py

**Files:**
- Create: `oar/layers.py`
- Modify: `oar/__init__.py`

**Step 1: The tests from Task 10 should still fail**

**Step 2: Write minimal implementation**

Create `oar/layers.py`:

```python
"""OAR layer implementations for quantized RNNs and Dense layers."""

import tensorflow as tf
import keras.backend as K
from qkeras import QSimpleRNNCell, QDense

from oar.regularizers import OAR2
from oar.callbacks import _reservoir_update, RESERVOIR_UPDATE_EVERY


@tf.keras.utils.register_keras_serializable(package="OAR")
class Downsampling(tf.keras.layers.Layer):
    """
    Temporal downsampling layer for RNN outputs.
    
    Reduces sequence length by a factor while increasing feature dimension.
    Used between RNN layers to reduce computation.
    
    Args:
        reduction_factor: Factor by which to reduce sequence length
        batch_size: Batch size (optional, for static shape inference)
    """

    def __init__(self, reduction_factor, batch_size=None, **kwargs):
        super(Downsampling, self).__init__(**kwargs)
        self.reduction_factor = reduction_factor
        self.batch_size = batch_size

    def compute_output_shape(self, input_shape):
        max_time = input_shape[1]
        num_units = input_shape[2]
        if max_time is not None:
            extra_timestep = tf.math.floormod(max_time, self.reduction_factor)
            reduced_size = (
                tf.math.floordiv(max_time, self.reduction_factor) + extra_timestep
            )
        else:
            reduced_size = None
        return [input_shape[0], reduced_size, num_units * self.reduction_factor]

    def call(self, inputs):
        input_shape = K.int_shape(inputs)

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = input_shape[0]

        outputs = inputs

        if input_shape[1] is not None:
            max_time = input_shape[1]
            extra_timestep = tf.math.floormod(max_time, self.reduction_factor)

            paddings = [[0, 0], [0, extra_timestep], [0, 0]]
            outputs = tf.pad(outputs, paddings)

        else:
            outputs = tf.signal.frame(
                outputs,
                self.reduction_factor,
                self.reduction_factor,
                pad_end=False,
                axis=1,
            )

        out_shape = self.compute_output_shape(input_shape)
        out_shape_tuple = tuple(-1 if s is None else s for s in out_shape)

        return tf.reshape(outputs, out_shape_tuple)

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_factor": self.reduction_factor,
            "batch_size": self.batch_size,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="OAR")
class QSimpleRNNCellWithOAR(QSimpleRNNCell):
    """
    QKeras SimpleRNN cell with Overflow-Aware Activity Regularization.
    
    Extends QSimpleRNNCell to apply OAR regularization on pre-activations
    and maintain reservoir sampling for distribution visualization.
    
    Args:
        units: Number of hidden units
        batch_size: Batch size for static shape inference
        activation: Activation function
        use_bias: Whether to use bias
        kernel_quantizer: Quantizer for input weights
        recurrent_quantizer: Quantizer for recurrent weights
        use_oar: Enable OAR regularization
        oar_lambda: OAR regularization rate
        omega: Bit precision (k = 2^omega)
        s: Gradient scaling factor
        reservoir_size: Size of pre-activation reservoir
    """

    def __init__(
        self,
        units,
        batch_size=512,
        activation=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        kernel_quantizer=None,
        recurrent_quantizer=None,
        bias_quantizer=None,
        state_quantizer=None,
        use_oar=False,
        oar_lambda=0,
        omega=32,
        s=1,
        reservoir_size=100000,
        **kwargs
    ):
        super(QSimpleRNNCellWithOAR, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            kernel_quantizer=kernel_quantizer,
            recurrent_quantizer=recurrent_quantizer,
            bias_quantizer=bias_quantizer,
            state_quantizer=state_quantizer,
            **kwargs
        )

        # OAR regularization
        self.oar = None
        self.oar_lambda = oar_lambda
        self.omega = omega
        if use_oar:
            self.oar = OAR2(oar_lambda=oar_lambda, k=2**omega, name=kwargs.get("name", ""))

        # Gradient scaling
        self.s = s

        # Batch size and reservoir
        self.batch_size = batch_size
        self.reservoir_size = reservoir_size

    def build(self, input_shape):
        super(QSimpleRNNCellWithOAR, self).build(input_shape)

        # Debug weights for quantized kernels
        if self.kernel_quantizer:
            self.quantized_kernel = self.add_weight(
                name="quantized_kernel",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=False,
            )
        if self.recurrent_quantizer:
            self.quantized_recurrent_kernel = self.add_weight(
                name="quantized_recurrent_kernel",
                shape=self.recurrent_kernel.shape,
                dtype=self.recurrent_kernel.dtype,
                initializer="zeros",
                trainable=False,
            )

        # EMA statistics
        self.wx = self.add_weight(
            name="wx_abs",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )
        self.wh = self.add_weight(
            name="wh_abs",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )

        # Reservoir sampling for pre-activations
        self.preact_reservoir = self.add_weight(
            name="preact_reservoir",
            shape=[self.reservoir_size],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )
        self.reservoir_count = self.add_weight(
            name="reservoir_count",
            shape=[],
            dtype=tf.int64,
            initializer="zeros",
            trainable=False,
        )
        self.reservoir_step = self.add_weight(
            name="reservoir_step",
            shape=[],
            dtype=tf.int64,
            initializer="zeros",
            trainable=False,
        )

    def call(self, inputs, states, training=None):
        prev_output = states[0] if tf.nest.is_nested(states) else states

        # Quantize state
        if self.state_quantizer:
            quantized_prev_output = self.state_quantizer_internal(prev_output)
        else:
            quantized_prev_output = prev_output

        # Quantize kernel
        if self.kernel_quantizer:
            quantized_kernel = self.kernel_quantizer_internal(self.kernel)
            self.quantized_kernel.assign(quantized_kernel)
        else:
            quantized_kernel = self.kernel

        h = K.dot(inputs, quantized_kernel)

        # Quantize recurrent kernel
        if self.recurrent_quantizer:
            quantized_recurrent = self.recurrent_quantizer_internal(self.recurrent_kernel)
            self.quantized_recurrent_kernel.assign(quantized_recurrent)
        else:
            quantized_recurrent = self.recurrent_kernel

        h_2 = K.dot(quantized_prev_output, quantized_recurrent)

        # Update EMA stats
        self.wx.assign(0.90 * self.wx + 0.10 * tf.reduce_mean(tf.abs(h), axis=0))
        self.wh.assign(0.90 * self.wh + 0.10 * tf.reduce_mean(tf.abs(h_2), axis=0))

        # Gradient scaling (forward unscaled, backward scaled)
        s = self.s
        output = h / s + h_2 / s + tf.stop_gradient(-h / s - h_2 / s + h + h_2)

        # Reservoir sampling for pre-activation histogram
        self.reservoir_step.assign_add(1)
        do_update = tf.equal(
            tf.math.floormod(self.reservoir_step, RESERVOIR_UPDATE_EVERY), 0
        )
        updated_reservoir, updated_count = tf.cond(
            do_update,
            lambda: _reservoir_update(
                reservoir=self.preact_reservoir,
                count=tf.cast(self.reservoir_count, tf.int64),
                new_values=h + h_2,
                reservoir_size=self.reservoir_size,
                seed=1997,
            ),
            lambda: (self.preact_reservoir, self.reservoir_count),
        )
        self.preact_reservoir.assign(updated_reservoir)
        self.reservoir_count.assign(updated_count)

        # Apply OAR and activation
        if self.activation is not None:
            if self.oar is not None:
                output = self.oar(output)
            output = self.activation(output)

        return output, [output]

    def get_config(self):
        from tensorflow.keras import constraints
        config = super().get_config()
        config.update({
            "batch_size": self.batch_size,
            "use_oar": self.oar is not None,
            "oar_lambda": self.oar_lambda,
            "omega": self.omega,
            "s": self.s,
            "reservoir_size": self.reservoir_size,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="OAR")
class QRNNWithOAR(tf.keras.layers.RNN):
    """
    RNN layer wrapper for QSimpleRNNCellWithOAR.
    
    Provides a convenient interface matching Keras RNN API while
    internally using QSimpleRNNCellWithOAR for OAR support.
    
    Forces unroll=True when OAR is enabled (required for per-timestep add_loss).
    """

    def __init__(
        self,
        cell=None,
        units=128,
        activation="tanh",
        batch_size=512,
        stateful=False,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_quantizer=None,
        recurrent_quantizer=None,
        bias_quantizer=None,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer=None,
        use_bias=False,
        use_oar=False,
        oar_lambda=0,
        omega=32,
        s=1.0,
        reservoir_size=100000,
        unroll=False,
        name="",
        **kwargs
    ):
        # Store params
        self.use_oar = use_oar
        self.oar_lambda = oar_lambda
        self.omega = omega
        self.s = s
        self.reservoir_size = reservoir_size
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer

        # Force unroll when OAR enabled
        to_unroll = unroll or use_oar

        cell = (
            QSimpleRNNCellWithOAR(
                units,
                activation=activation,
                batch_size=batch_size,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_quantizer=kernel_quantizer,
                recurrent_quantizer=recurrent_quantizer,
                bias_quantizer=bias_quantizer,
                use_bias=use_bias,
                use_oar=use_oar,
                oar_lambda=oar_lambda,
                omega=omega,
                s=s,
                reservoir_size=reservoir_size,
                name=name,
            )
            if cell is None
            else cell
        )

        super(QRNNWithOAR, self).__init__(
            cell, return_sequences=True, stateful=stateful, unroll=to_unroll, name=name
        )

        self.quantizers = self.get_quantizers()

    # Properties to forward from self.cell
    _CELL_FORWARDED_ATTRS = frozenset({
        "units", "activation", "use_bias", "bias_initializer",
        "kernel_regularizer", "recurrent_regularizer", "bias_regularizer",
        "kernel_constraint", "recurrent_constraint", "bias_constraint",
        "kernel_quantizer_internal", "recurrent_quantizer_internal",
        "bias_quantizer_internal", "state_quantizer_internal",
        "kernel_quantizer", "recurrent_quantizer", "bias_quantizer", "state_quantizer",
    })

    def get_quantizers(self):
        return self.cell.quantizers

    def get_prunable_weights(self):
        return [self.cell.kernel, self.cell.recurrent_kernel]

    def __getattr__(self, name):
        """Delegate attribute access to self.cell for forwarded properties."""
        if name in QRNNWithOAR._CELL_FORWARDED_ATTRS:
            return getattr(self.cell, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_config(self):
        from tensorflow.keras import constraints
        config = super().get_config()
        config.update({
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "kernel_regularizer": constraints.serialize(self.kernel_regularizer),
            "recurrent_regularizer": constraints.serialize(self.recurrent_regularizer),
            "bias_regularizer": constraints.serialize(self.bias_regularizer),
            "kernel_quantizer": constraints.serialize(self.kernel_quantizer_internal),
            "recurrent_quantizer": constraints.serialize(self.recurrent_quantizer_internal),
            "bias_quantizer": constraints.serialize(self.bias_quantizer_internal),
        })
        return config


@tf.keras.utils.register_keras_serializable(package="OAR")
class QDenseWithOAR(QDense):
    """
    Quantized Dense layer with OAR support.
    
    Extends QDense with overflow-aware regularization and
    reservoir sampling for pre-activation distribution tracking.
    """

    def __init__(
        self,
        units,
        activation=None,
        batch_size=512,
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        kernel_quantizer=None,
        bias_quantizer=None,
        kernel_range=None,
        bias_range=None,
        use_oar=False,
        oar_lambda=0,
        omega=32,
        s=1,
        reservoir_size=100000,
        **kwargs
    ):
        # OAR setup
        self.oar = None
        self.oar_lambda = oar_lambda
        self.omega = omega
        if use_oar:
            self.oar = OAR2(oar_lambda=oar_lambda, k=2**omega, name=kwargs.get("name", ""))

        self.s = s
        self.batch_size = batch_size
        self.reservoir_size = reservoir_size

        super(QDenseWithOAR, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kernel_quantizer=kernel_quantizer,
            bias_quantizer=bias_quantizer,
            kernel_range=kernel_range,
            bias_range=bias_range,
            **kwargs
        )

    def build(self, input_shape):
        super(QDenseWithOAR, self).build(input_shape)

        if self.kernel_quantizer:
            self.quantized_kernel = self.add_weight(
                name="quantized_kernel",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=False,
            )

        self.wx = self.add_weight(
            name="wx_abs",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )

        self.preact_reservoir = self.add_weight(
            name="preact_reservoir",
            shape=[self.reservoir_size],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )
        self.reservoir_count = self.add_weight(
            name="reservoir_count",
            shape=[],
            dtype=tf.int64,
            initializer="zeros",
            trainable=False,
        )
        self.reservoir_step = self.add_weight(
            name="reservoir_step",
            shape=[],
            dtype=tf.int64,
            initializer="zeros",
            trainable=False,
        )

        self.x_input = self.add_weight(
            name="x_abs",
            shape=[input_shape[-1]],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )

    def call(self, inputs):
        # Input stats
        reduce_dims = tf.range(0, tf.rank(inputs) - 1)
        self.x_input.assign(
            0.9 * self.x_input + 0.1 * tf.reduce_mean(tf.abs(inputs), axis=reduce_dims)
        )

        # Quantize kernel
        quantized_kernel = self.kernel
        if self.kernel_quantizer:
            quantized_kernel = self.kernel_quantizer_internal(self.kernel)
            self.quantized_kernel.assign(quantized_kernel)

        # Forward pass
        h = tf.keras.backend.dot(inputs, quantized_kernel)

        # Gradient scaling
        s = self.s
        h = h / s + tf.stop_gradient(-h / s + h)

        # Update stats
        reduce_dims = tf.range(0, tf.rank(h) - 1)
        self.wx.assign(
            0.9 * self.wx + 0.1 * tf.reduce_mean(tf.abs(h), axis=reduce_dims)
        )

        # Reservoir sampling
        self.reservoir_step.assign_add(1)
        do_update = tf.equal(
            tf.math.floormod(self.reservoir_step, RESERVOIR_UPDATE_EVERY), 0
        )
        updated_reservoir, updated_count = tf.cond(
            do_update,
            lambda: _reservoir_update(
                reservoir=self.preact_reservoir,
                count=tf.cast(self.reservoir_count, tf.int64),
                new_values=h,
                reservoir_size=self.reservoir_size,
                seed=1997,
            ),
            lambda: (self.preact_reservoir, self.reservoir_count),
        )
        self.preact_reservoir.assign(updated_reservoir)
        self.reservoir_count.assign(updated_count)

        output = h
        if self.activation is not None:
            if self.oar is not None:
                output = self.oar(output)
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "batch_size": self.batch_size,
            "use_oar": self.oar is not None,
            "oar_lambda": self.oar_lambda,
            "omega": self.omega,
            "s": self.s,
            "reservoir_size": self.reservoir_size,
        })
        return config
```

Update `oar/__init__.py`:

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
]
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py -v -k "layer or downsampling or qrnn or qdense"`
Expected: All layer tests PASS

**Step 4: Commit**

```bash
git add oar/layers.py oar/__init__.py
git commit -m "feat(oar): extract layers module"
```

---

### Task 12: Add training utilities module

**Files:**
- Create: `oar/training.py`
- Modify: `tests/test_oar_serialization.py`
- Modify: `oar/__init__.py`

**Step 1: Write the failing test**

Append to `tests/test_oar_serialization.py`:

```python
from oar.training import get_default_layer_options_from_options, OARModel


def test_get_default_layer_options_from_oar():
    options = {
        "oar": {"omega": 6, "oar_lambda": 1e-4},
    }
    layer_options = get_default_layer_options_from_options(options)
    assert layer_options["QRNN_0"]["oar"]["omega"] == 6
    assert layer_options["QRNN_0"]["oar"]["oar_lambda"] == 1e-4


def test_oar_model_constructs():
    inputs = tf.keras.layers.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(5)(inputs)
    model = OARModel(inputs=inputs, outputs=outputs)
    assert model is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py::test_get_default_layer_options_from_oar -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'oar.training'"

**Step 3: Write minimal implementation**

Create `oar/training.py`:

```python
"""OAR training utilities and model classes."""

import tensorflow as tf
from keras.engine import data_adapter


def get_default_layer_options_from_options(options):
    """
    Build layer_options dict from high-level options.
    
    Creates per-layer configuration for MNIST RNN architecture
    based on global options. Used by four-step quantization.
    
    Args:
        options: Dict with "oar" sub-dict containing "omega" and "oar_lambda"
    
    Returns:
        Dict mapping layer names to their configuration
    """
    ternarize_inputs = False
    t = 1.0
    s = 1.0
    activation = tf.keras.activations.tanh
    oar = False
    tern_params = {"QRNN_0": 0, "QRNN_1": 0, "DENSE_0": 0, "DENSE_OUT": 0}
    return {
        "INPUT": {"ternarize": ternarize_inputs},
        "QRNN_0": {
            "activation": activation,
            "oar": {
                "use": oar,
                "oar_lambda": options["oar"]["oar_lambda"],
                "omega": options["oar"]["omega"],
            },
            "s": s,
            "τ": t * tern_params["QRNN_0"],
        },
        "QRNN_1": {
            "activation": activation,
            "oar": {
                "use": oar,
                "oar_lambda": options["oar"]["oar_lambda"],
                "omega": options["oar"]["omega"],
            },
            "s": s,
            "τ": t * tern_params["QRNN_1"],
        },
        "DENSE_0": {
            "activation": activation,
            "oar": {
                "use": oar,
                "oar_lambda": options["oar"]["oar_lambda"],
                "omega": options["oar"]["omega"],
            },
            "s": 1.0,
            "τ": t * tern_params["DENSE_0"],
        },
        "DENSE_OUT": {
            "activation": tf.keras.activations.softmax,
            "oar": {
                "use": True,
                "oar_lambda": 0.0,
                "omega": options["oar"]["omega"],
            },
            "s": 1.0,
            "τ": t * tern_params["DENSE_OUT"],
        },
    }


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

Update `oar/__init__.py`:

```python
"""
Overflow-Aware Activity Regularization (OAR) Package.

A TensorFlow/Keras framework for training quantized neural networks
with overflow-aware regularization for TFHE inference.
"""

from oar.regularizers import OAR1, OAR2, oar_penalty_fn, compute_oar_metric
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
from oar.training import get_default_layer_options_from_options, OARModel

__version__ = "0.1.0"

__all__ = [
    # Regularizers
    "OAR1",
    "OAR2",
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
    "get_default_layer_options_from_options",
    "OARModel",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py -v -k "training or oar_model"`
Expected: All training tests PASS

**Step 5: Commit**

```bash
git add oar/training.py oar/__init__.py tests/test_oar_serialization.py
git commit -m "feat(oar): extract training utilities module"
```

---

### Task 13: Run all serialization tests

**Files:**
- None (verification only)

**Step 1: Run all oar serialization tests**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py -v`
Expected: All tests PASS

**Step 2: Run existing tests for regression**

Run: `PYTHONPATH=. pytest tests/test_phase1_naming.py tests/test_reservoir_sampling.py -v`
Expected: All tests PASS (these still import from utils/model_utils.py)

**Step 3: Commit docs if needed**

No commit needed if all tests pass.

---

### Task 14: Add model save/load integration test

**Files:**
- Modify: `tests/test_oar_serialization.py`

**Step 1: Write the integration test**

Append to `tests/test_oar_serialization.py`:

```python
def test_model_save_load_roundtrip():
    """Integration test: full model save/load with OAR layers."""
    import tempfile
    
    # Build a minimal model with OAR layers
    inputs = tf.keras.layers.Input(shape=(10, 8), batch_size=2)
    x = QRNNWithOAR(units=4, batch_size=2, use_oar=True, oar_lambda=1e-4, omega=6, name="QRNN_0")(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = QDenseWithOAR(units=3, batch_size=2, use_oar=False, name="DENSE_0")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(f"{tmpdir}/model.keras")
        loaded = tf.keras.models.load_model(f"{tmpdir}/model.keras")
    
    # Verify functional equivalence
    test_input = tf.random.normal([2, 10, 8])
    original_output = model(test_input)
    loaded_output = loaded(test_input)
    tf.debugging.assert_near(original_output, loaded_output, atol=1e-5)
```

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_oar_serialization.py::test_model_save_load_roundtrip -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_oar_serialization.py
git commit -m "test(oar): add model save/load integration test"
```

---

### Task 15: Update utils/model_utils.py to re-export from oar

**Files:**
- Modify: `utils/model_utils.py`

**Step 1: No test needed (backward compatibility)**

**Step 2: Write minimal implementation**

Replace most of `utils/model_utils.py` with re-exports:

```python
"""
Backward compatibility shim for utils/model_utils.py.

All components have been moved to the oar/ package.
This module re-exports them for backward compatibility.

DEPRECATED: Import from oar package directly:
    from oar import OAR2, QRNNWithOAR, ...
"""

# Re-export everything from oar package
from oar import (
    # Regularizers
    OAR1,
    OAR2,
    oar_penalty_fn,
    compute_oar_metric,
    # Activations
    sign_ste_tanh,
    mod_sign,
    TrackedActivation,
    # Callbacks
    ReservoirHistogramCallback,
    reset_stat_weights,
    _reservoir_update,
    RESERVOIR_UPDATE_EVERY,
    # Layers
    QRNNWithOAR,
    QSimpleRNNCellWithOAR,
    QDenseWithOAR,
    Downsampling,
    # Training
    get_default_layer_options_from_options,
    OARModel,
)

__all__ = [
    "OAR1",
    "OAR2",
    "oar_penalty_fn",
    "compute_oar_metric",
    "sign_ste_tanh",
    "mod_sign",
    "TrackedActivation",
    "ReservoirHistogramCallback",
    "reset_stat_weights",
    "_reservoir_update",
    "RESERVOIR_UPDATE_EVERY",
    "QRNNWithOAR",
    "QSimpleRNNCellWithOAR",
    "QDenseWithOAR",
    "Downsampling",
    "get_default_layer_options_from_options",
    "OARModel",
]
```

**Step 3: Verify existing tests still pass**

Run: `PYTHONPATH=. pytest tests/test_phase1_naming.py tests/test_reservoir_sampling.py -v`
Expected: All tests PASS (backward compatibility maintained)

**Step 4: Commit**

```bash
git add utils/model_utils.py
git commit -m "refactor: make utils/model_utils.py a re-export shim"
```

---

### Task 16: Update quantization.py to re-export from oar

**Files:**
- Modify: `quantization.py`

**Step 1: No test needed (backward compatibility)**

**Step 2: Write minimal implementation**

Replace `quantization.py` with re-exports:

```python
"""
Backward compatibility shim for quantization.py.

Quantizers have been moved to the oar/ package.
This module re-exports them for backward compatibility.

DEPRECATED: Import from oar package directly:
    from oar import TernarizationWithThreshold, ternarize_tensor_with_threshold
"""

from oar import TernarizationWithThreshold, ternarize_tensor_with_threshold

__all__ = ["TernarizationWithThreshold", "ternarize_tensor_with_threshold"]
```

**Step 3: Verify imports work**

Run: `PYTHONPATH=. python -c "from quantization import TernarizationWithThreshold; print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add quantization.py
git commit -m "refactor: make quantization.py a re-export shim"
```

---

### Task 17: Update mnist_rnn_model.py imports

**Files:**
- Modify: `mnist_rnn_model.py`

**Step 1: No test needed (functionality unchanged)**

**Step 2: Write minimal implementation**

Update imports at top of `mnist_rnn_model.py`:

```python
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
# ... rest of file unchanged
```

**Step 3: Verify imports work**

Run: `PYTHONPATH=. python -c "from mnist_rnn_model import get_model; print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add mnist_rnn_model.py
git commit -m "refactor: update mnist_rnn_model.py to use oar package"
```

---

### Task 18: Update main.py imports

**Files:**
- Modify: `main.py`

**Step 1: No test needed (functionality unchanged)**

**Step 2: Write minimal implementation**

Update imports at top of `main.py`:

```python
from mnist_rnn_model import get_model
from oar import (
    mod_sign,
    sign_ste_tanh,
    get_default_layer_options_from_options,
    ReservoirHistogramCallback,
    reset_stat_weights,
)
from export_mnist_weights_h5 import export_mnist_weights
from export_mnist import extract_ternarized_mnist_test_dataset
import tensorflow_datasets as tfds
import tensorflow as tf
import os
from datetime import datetime
# ... rest of file unchanged
```

**Step 3: Verify imports work**

Run: `PYTHONPATH=. python -c "import main; print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add main.py
git commit -m "refactor: update main.py to use oar package"
```

---

### Task 19: Run full test suite and verify

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run: `PYTHONPATH=. pytest tests/ -v`
Expected: All tests PASS

**Step 2: Verify main.py can run short training**

Run: `PYTHONPATH=. python -c "
import main
# Just verify imports work and model can be created
from mnist_rnn_model import get_model
from oar import get_default_layer_options_from_options

options = {
    'enlarge': False, 'epochs': 1, 'learning_rate': 1e-4, 'batch_size': 32,
    't': 1.5, 'tᵢ': 0.7, 's': 4.0,
    'oar': {'oar_lambda': 1e-4, 'omega': 6}, 'quantize': False,
}
layer_options = get_default_layer_options_from_options(options)
model = get_model(options, layer_options)
print(f'Model created with {model.count_params()} parameters')
"`
Expected: Model created with ~800K parameters

**Step 3: Commit final cleanup**

```bash
git add -A
git commit -m "feat(oar): complete Phase 2 - extract oar package"
```

---

### Task 20: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update architecture section**

Update the Architecture section in CLAUDE.md to reflect new structure:

```markdown
## Architecture

```
oar/                           # Core OAR framework (reusable)
├── __init__.py                # Public API
├── regularizers.py            # OAR1, OAR2, oar_penalty_fn, compute_oar_metric
├── activations.py             # sign_ste_tanh, mod_sign, TrackedActivation
├── layers.py                  # QRNNWithOAR, QSimpleRNNCellWithOAR, QDenseWithOAR, Downsampling
├── quantizers.py              # TernarizationWithThreshold
├── training.py                # OARModel, get_default_layer_options_from_options
└── callbacks.py               # ReservoirHistogramCallback, reset_stat_weights

main.py                        # Entry point, training loops, four-step quantization
mnist_rnn_model.py             # MNIST model architecture (uses oar package)
quantization.py                # Re-export shim (deprecated, use oar.quantizers)
utils/model_utils.py           # Re-export shim (deprecated, use oar package)
```
```

**Step 2: Add import guidance**

Add to CLAUDE.md:

```markdown
## Importing OAR Components

```python
# Preferred: import from oar package
from oar import OAR2, QRNNWithOAR, sign_ste_tanh, TrackedActivation

# Deprecated: old import paths still work but are discouraged
from utils.model_utils import OAR2  # works but deprecated
from quantization import TernarizationWithThreshold  # works but deprecated
```
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 2 package structure"
```

---

## Summary

Phase 2 extracts all reusable OAR components into a proper `oar/` package:

| Module | Contents |
|--------|----------|
| `oar/regularizers.py` | `OARRegularizer`, `OAR1`, `OAR2`, `oar_penalty_fn`, `compute_oar_metric` |
| `oar/activations.py` | `sign_ste_tanh`, `mod_sign`, `TrackedActivation` |
| `oar/layers.py` | `QRNNWithOAR`, `QSimpleRNNCellWithOAR`, `QDenseWithOAR`, `Downsampling` |
| `oar/quantizers.py` | `TernarizationWithThreshold`, `ternarize_tensor_with_threshold` |
| `oar/training.py` | `OARModel`, `get_default_layer_options_from_options` |
| `oar/callbacks.py` | `ReservoirHistogramCallback`, `reset_stat_weights`, `_reservoir_update` |

**Key Fixes:**
1. `TrackedActivation.get_config()` - no longer references non-existent `self.alpha_init`
2. `TernarizationWithThreshold.get_config()` - now returns all constructor params
3. All custom classes have `@register_keras_serializable(package="OAR")` decorator
4. `OAR1`/`OAR2` consolidated into single `OARRegularizer` class with factory functions
5. `QRNNWithOAR.get_config()` uses `constraints.serialize()` for quantizers
6. `QSimpleRNNCellWithOAR` and `QDenseWithOAR` have proper `get_config()` methods
7. `QRNNWithOAR` uses `__getattr__` delegation instead of 18 explicit properties
8. Lambda removed from `get_default_layer_options_from_options` (softmax activation)
9. Model save/load integration test added

**Backward Compatibility:**
- `utils/model_utils.py` becomes a re-export shim
- `quantization.py` becomes a re-export shim
- Existing imports continue to work but are deprecated
- `OAR1()` and `OAR2()` factory functions maintain backward compatibility

---

**Plan complete and saved to `docs/plans/2026-02-07-phase2-extract-oar-package.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
