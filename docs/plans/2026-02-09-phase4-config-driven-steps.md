# Phase 4: Configuration-Driven Quantization Steps

> **For Agent:** REQUIRED: Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Replace hardcoded step logic with declarative configuration. Enable N-step quantization for different experiments without code changes.

**Architecture:** Define a flat `LayerConfig` dataclass and `TrainingStepConfig` that contains all training parameters. Eliminate conversion layers—`get_model()` and `train()` consume `TrainingStepConfig` directly. Move MNIST-specific step definitions to `experiments/mnist/steps.py`.

**Tech Stack:** Python 3.10, TensorFlow 2.10.1, dataclasses

**Context:** Phase 4 of the OAR refactoring plan (see `docs/plans/2026-02-07-refactor-for-experiments.md`). Phase 3.5 complete.

**Evaluation Status:** Evaluated 2026-02-09. Simplified based on evaluation:
- Removed `build_layer_options()` conversion layer
- Flattened nested dataclasses (`OARConfig`, `TernaryQuantizationConfig` → inline in `LayerConfig`)
- Merged `options` dict into `TrainingStepConfig`
- No backwards compatibility needed

---

## Design Principles

1. **No conversion layer** - `get_model()` and `train()` consume `TrainingStepConfig` directly
2. **Flat config** - No nested dataclasses; all layer params are direct fields
3. **Presence = enabled** - `oar_lambda: float | None` where `None` = disabled
4. **Explicit per-layer** - Each step declares exactly what each layer does
5. **MNIST-specific in MNIST folder** - Layer names defined in `experiments/mnist/`, not `oar/`

---

## Dataclasses (in `oar/config.py`)

```python
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class LayerConfig:
    """Configuration for a single layer.
    
    Attributes:
        activation: Activation function name ("tanh", "sign_ste_tanh", "mod_sign", "softmax")
        gradient_scale: s - gradient scaling factor
        oar_lambda: OAR regularization rate (None = disabled)
        omega: Bit precision for OAR (k = 2^omega)
        quantize_threshold: Ternarization threshold τ (None = no quantization)
    """
    activation: str = "tanh"
    gradient_scale: float = 1.0
    oar_lambda: float | None = None
    omega: int = 6
    quantize_threshold: float | None = None
    
    def __post_init__(self):
        valid_activations = {"tanh", "sign_ste_tanh", "mod_sign", "softmax"}
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, got '{self.activation}'")
        if self.gradient_scale <= 0:
            raise ValueError(f"gradient_scale must be positive, got {self.gradient_scale}")
        if self.oar_lambda is not None and self.oar_lambda < 0:
            raise ValueError(f"oar_lambda must be non-negative, got {self.oar_lambda}")
        if self.omega < 1:
            raise ValueError(f"omega must be positive, got {self.omega}")
        if self.quantize_threshold is not None and self.quantize_threshold < 0:
            raise ValueError(f"quantize_threshold must be non-negative, got {self.quantize_threshold}")


@dataclass
class InputConfig:
    """Configuration for input ternarization.
    
    Attributes:
        quantize_threshold: Ternarization threshold (None = no quantization)
    """
    quantize_threshold: float | None = None
    
    def __post_init__(self):
        if self.quantize_threshold is not None and self.quantize_threshold < 0:
            raise ValueError(f"quantize_threshold must be non-negative, got {self.quantize_threshold}")


@dataclass 
class TrainingStepConfig:
    """Configuration for a single training step.
    
    Contains all parameters needed for training: hyperparameters and per-layer config.
    
    Attributes:
        name: Human-readable step name (for logging)
        epochs: Training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        enlarge: Use enlarged input (128x128) vs standard (28x28)
        layers: Per-layer configuration dict
        input_config: Input layer configuration
    """
    name: str
    epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 512
    enlarge: bool = False
    layers: dict[str, LayerConfig] = field(default_factory=dict)
    input_config: InputConfig = field(default_factory=InputConfig)
    
    def __post_init__(self):
        if self.epochs < 0:
            raise ValueError(f"epochs must be non-negative, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


def resolve_activation(name: str, omega: int = 6) -> Callable:
    """Resolve activation name to callable.
    
    Args:
        name: Activation name
        omega: Bit precision for mod_sign
        
    Returns:
        Activation function callable
    """
    import tensorflow as tf
    from functools import partial
    from oar import sign_ste_tanh, mod_sign
    
    if name == "tanh":
        return tf.keras.activations.tanh
    elif name == "sign_ste_tanh":
        return sign_ste_tanh
    elif name == "mod_sign":
        return partial(mod_sign, num_bits=omega)
    elif name == "softmax":
        return tf.keras.activations.softmax
    else:
        raise ValueError(f"Unknown activation: {name}")
```

---

## MNIST Steps (in `experiments/mnist/steps.py`)

```python
"""MNIST four-step quantization configuration."""

from oar.config import LayerConfig, InputConfig, TrainingStepConfig


def get_default_layer_config(layer_name: str) -> LayerConfig:
    """Get default LayerConfig for a layer.
    
    DENSE_OUT uses softmax, all others use tanh.
    """
    if layer_name == "DENSE_OUT":
        return LayerConfig(activation="softmax")
    return LayerConfig()


def compute_thresholds(
    tern_params: dict[str, float],
    t: float = 1.5,
) -> dict[str, float]:
    """Compute ternarization thresholds: τ = t × E[|θ|].
    
    Args:
        tern_params: Mean absolute weights per layer
        t: Threshold multiplier
        
    Returns:
        Thresholds per layer
    """
    return {name: t * mean_abs for name, mean_abs in tern_params.items()}


# Step 1: Tanh baseline - all defaults
STEP_1 = TrainingStepConfig(
    name="tanh_baseline",
    epochs=100,
)

# Step 2: Sign activation with gradient scaling for RNNs
STEP_2 = TrainingStepConfig(
    name="sign_with_gradient_scaling",
    epochs=1000,
    layers={
        "QRNN_0": LayerConfig(activation="sign_ste_tanh", gradient_scale=4.0),
        "QRNN_1": LayerConfig(activation="sign_ste_tanh", gradient_scale=4.0),
        "DENSE_0": LayerConfig(activation="sign_ste_tanh"),
    },
)

# Step 3: Quantize inputs
STEP_3 = TrainingStepConfig(
    name="quantize_inputs",
    epochs=1000,
    input_config=InputConfig(quantize_threshold=0.7),
    layers={
        "QRNN_0": LayerConfig(activation="sign_ste_tanh", gradient_scale=4.0),
        "QRNN_1": LayerConfig(activation="sign_ste_tanh", gradient_scale=4.0),
        "DENSE_0": LayerConfig(activation="sign_ste_tanh"),
    },
)


def create_step_4(tern_params: dict[str, float], t: float = 1.5) -> TrainingStepConfig:
    """Create Step 4 config with computed thresholds.
    
    Step 4 needs runtime threshold computation: τ = t × E[|θ|]
    
    Args:
        tern_params: Mean absolute weights per layer from previous step
        t: Threshold multiplier (default 1.5)
        
    Returns:
        TrainingStepConfig for step 4
    """
    thresholds = compute_thresholds(tern_params, t)
    
    return TrainingStepConfig(
        name="full_quantization_with_oar",
        epochs=1000,
        learning_rate=1e-5,
        input_config=InputConfig(quantize_threshold=0.7),
        layers={
            "QRNN_0": LayerConfig(
                activation="mod_sign",
                gradient_scale=4.0,
                oar_lambda=1e-4,
                omega=6,
                quantize_threshold=thresholds.get("QRNN_0", 0),
            ),
            "QRNN_1": LayerConfig(
                activation="mod_sign",
                gradient_scale=4.0,
                oar_lambda=1e-4,
                omega=6,
                quantize_threshold=thresholds.get("QRNN_1", 0),
            ),
            "DENSE_0": LayerConfig(
                activation="mod_sign",
                oar_lambda=1e-4,
                omega=6,
                quantize_threshold=thresholds.get("DENSE_0", 0),
            ),
            "DENSE_OUT": LayerConfig(
                activation="softmax",
                oar_lambda=0.0,  # Track stats only
                omega=6,
                quantize_threshold=thresholds.get("DENSE_OUT", 0),
            ),
        },
    )


# Static steps (1-3) that don't need runtime computation
STATIC_STEPS = [STEP_1, STEP_2, STEP_3]
```

---

## Updated `get_model()` (in `experiments/mnist/model.py`)

The key change: `get_model()` now takes `TrainingStepConfig` directly.

```python
def get_model(step_config: TrainingStepConfig) -> OARModel:
    """Build MNIST RNN model from step configuration.
    
    Args:
        step_config: Training step configuration
        
    Returns:
        Compiled OARModel
    """
    from oar import (
        QRNNWithOAR, QDenseWithOAR, Downsampling,
        TrackedActivation, TernarizationWithThreshold,
        OARModel, ternarize_tensor_with_threshold,
    )
    from oar.config import resolve_activation
    from experiments.mnist.steps import get_default_layer_config
    
    def get_layer_cfg(name: str) -> LayerConfig:
        """Get config for layer, using defaults if not specified."""
        return step_config.layers.get(name, get_default_layer_config(name))
    
    def make_quantizer(threshold: float | None):
        """Create quantizer if threshold is set."""
        if threshold is None:
            return None
        return TernarizationWithThreshold(threshold=threshold)
    
    # Input shape based on enlarge setting
    if step_config.enlarge:
        input_shape = (128, 128, 1)
        seq_len, features = 128, 128
        model_name = "ENLARGED_MNIST_RNN"
    else:
        input_shape = (28, 28, 1)
        seq_len, features = 28, 28
        model_name = "MNIST_RNN"
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Input ternarization
    if step_config.input_config.quantize_threshold is not None:
        theta = step_config.input_config.quantize_threshold
        x = tf.keras.layers.Lambda(
            lambda x: ternarize_tensor_with_threshold(x, theta),
            name="TERNARIZE_INPUT",
        )(inputs)
    else:
        x = tf.keras.layers.Lambda(lambda x: x, name="NOOP")(inputs)
    
    x = tf.keras.layers.Reshape((seq_len, features))(x)
    
    # QRNN_0
    cfg = get_layer_cfg("QRNN_0")
    x = QRNNWithOAR(
        units=128,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation, cfg.omega),
            name="QRNN_0",
        ),
        batch_size=step_config.batch_size,
        use_oar=cfg.oar_lambda is not None,
        oar_lambda=cfg.oar_lambda or 0.0,
        omega=cfg.omega,
        s=cfg.gradient_scale,
        kernel_quantizer=make_quantizer(cfg.quantize_threshold),
        recurrent_quantizer=make_quantizer(cfg.quantize_threshold),
        return_sequences=True,
        name="QRNN_0",
    )(x)
    
    x = Downsampling(reduction_factor=2, batch_size=step_config.batch_size)(x)
    
    # QRNN_1
    cfg = get_layer_cfg("QRNN_1")
    x = QRNNWithOAR(
        units=128,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation, cfg.omega),
            name="QRNN_1",
        ),
        batch_size=step_config.batch_size,
        use_oar=cfg.oar_lambda is not None,
        oar_lambda=cfg.oar_lambda or 0.0,
        omega=cfg.omega,
        s=cfg.gradient_scale,
        kernel_quantizer=make_quantizer(cfg.quantize_threshold),
        recurrent_quantizer=make_quantizer(cfg.quantize_threshold),
        return_sequences=False,
        name="QRNN_1",
    )(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    # DENSE_0
    cfg = get_layer_cfg("DENSE_0")
    x = QDenseWithOAR(
        units=1024,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation, cfg.omega),
            name="DENSE_0",
        ),
        batch_size=step_config.batch_size,
        use_oar=cfg.oar_lambda is not None,
        oar_lambda=cfg.oar_lambda or 0.0,
        omega=cfg.omega,
        s=cfg.gradient_scale,
        kernel_quantizer=make_quantizer(cfg.quantize_threshold),
        name="DENSE_0",
    )(x)
    
    # DENSE_OUT
    cfg = get_layer_cfg("DENSE_OUT")
    outputs = QDenseWithOAR(
        units=10,
        activation=TrackedActivation(
            activation=resolve_activation(cfg.activation, cfg.omega),
            name="DENSE_OUT",
        ),
        batch_size=step_config.batch_size,
        use_oar=cfg.oar_lambda is not None,
        oar_lambda=cfg.oar_lambda or 0.0,
        omega=cfg.omega,
        s=cfg.gradient_scale,
        kernel_quantizer=make_quantizer(cfg.quantize_threshold),
        name="DENSE_OUT",
    )(x)
    
    return OARModel(inputs=inputs, outputs=outputs, name=model_name)
```

---

## Updated `train()` (in `experiments/mnist/training.py`)

```python
def train(
    step_config: TrainingStepConfig,
    step_number: int | None = None,
    pretrained_weights: str | None = None,
    run_dir: str | None = None,
) -> str:
    """Run training for a step.
    
    Args:
        step_config: Training step configuration (contains all params)
        step_number: Step number for logging (1-indexed)
        pretrained_weights: Path to checkpoints folder
        run_dir: Run directory (creates new if None)
        
    Returns:
        Path to checkpoints folder
    """
    from experiments.mnist.model import get_model
    from experiments.mnist.data import get_datasets
    
    strategy, _ = configure_environment()
    
    if run_dir is None:
        run_dir = create_run_dir()
    
    if step_number is not None:
        output_dir = os.path.join(run_dir, f"step_{step_number}") + "/"
    else:
        output_dir = run_dir
    
    os.makedirs(output_dir, exist_ok=True)
    _save_step_config(output_dir, step_number, step_config, pretrained_weights)
    
    ds_train, ds_val, ds_test = get_datasets(
        batch_size=step_config.batch_size,
        enlarge=step_config.enlarge,
    )
    
    with strategy.scope():
        model = get_model(step_config)
        
        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)
            print(f"Restored pretrained weights from {pretrained_weights}.")
        
        reset_stat_weights(model)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=step_config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
    
    # Callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=output_dir + TB_LOGS_DIR,
        histogram_freq=1,
        update_freq="epoch",
    )
    reservoir_cb = ReservoirHistogramCallback(log_dir=output_dir + TB_LOGS_DIR)
    
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        tf.keras.optimizers.schedules.CosineDecay(
            step_config.learning_rate, 100, alpha=0.1
        ),
        verbose=0,
    )
    
    ckpt_callback = None
    if RECORD_CKPTS:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir + CKPT_DIR,
            save_weights_only=True,
            save_best_only=False,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        )
    
    model.fit(
        ds_train,
        epochs=step_config.epochs,
        validation_data=ds_val,
        callbacks=[cb for cb in [tb_callback, reservoir_cb, ckpt_callback, lr_callback] if cb],
        verbose=2,
    )
    
    print("\nRUNNING EVALUATION OVER TEST SET\n")
    model.evaluate(ds_test, verbose=2)
    
    return output_dir + CKPT_DIR
```

---

## Updated Training Orchestration (in `experiments/mnist/config.py`)

```python
"""MNIST experiment configuration and training orchestration."""

import json
import os
from dataclasses import asdict
from datetime import datetime

from oar.config import TrainingStepConfig


def get_model_parameter_stats(
    pretrained_weights: str | None,
    step_config: TrainingStepConfig,
    layer_names: list[str] | None = None,
) -> dict[str, float]:
    """Get mean absolute weights per layer for threshold computation.
    
    Args:
        pretrained_weights: Path to checkpoints folder
        step_config: Step config (used to build model for loading weights)
        layer_names: Layer names to compute stats for (default: MNIST layers)
        
    Returns:
        Dict mapping layer names to mean absolute weights
    """
    import tensorflow as tf
    from experiments.mnist.model import get_model
    
    # Default to MNIST layers if not specified
    if layer_names is None:
        layer_names = ["QRNN_0", "QRNN_1", "DENSE_0", "DENSE_OUT"]
    
    tern_params = {name: 0.0 for name in layer_names}
    
    if pretrained_weights is None:
        return tern_params
    
    model = get_model(step_config)
    model.load_weights(pretrained_weights)
    print(f"Loaded weights from {pretrained_weights} for threshold computation.")
    
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:
            all_weights = tf.concat(
                [tf.reshape(w, [-1]) for w in layer.trainable_weights], axis=-1
            )
            mean_abs = float(tf.reduce_mean(tf.abs(all_weights)).numpy())
            
            for name in layer_names:
                if name in layer.name:
                    tern_params[name] = mean_abs
                    break
    
    return tern_params


def _save_experiment_config(run_dir: str, steps: list[TrainingStepConfig]) -> None:
    """Save experiment configuration."""
    config = {
        "started_at": datetime.now().isoformat(),
        "experiment_type": "four_step_quantization",
        "steps": [asdict(s) for s in steps],
    }
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def perform_training_steps(
    steps: list[TrainingStepConfig],
    step_4_factory=None,
) -> str:
    """Execute training steps.
    
    Args:
        steps: List of static TrainingStepConfig (typically steps 1-3)
        step_4_factory: Optional factory function(tern_params) -> TrainingStepConfig
            for steps requiring runtime threshold computation
        
    Returns:
        Path to final checkpoint folder
    """
    from experiments.mnist.training import train, create_run_dir, tee_output
    from experiments.mnist.steps import create_step_4
    
    if not steps:
        raise ValueError("steps list cannot be empty")
    
    if step_4_factory is None:
        step_4_factory = create_step_4
    
    run_dir = create_run_dir()
    log_path = os.path.join(run_dir, "output.log")
    
    with tee_output(log_path):
        pretrained_weights = None
        
        for i, step_config in enumerate(steps, start=1):
            print(f"\nPERFORMING STEP {i}/{len(steps) + (1 if step_4_factory else 0)}: {step_config.name}\n")
            
            pretrained_weights = train(
                step_config=step_config,
                step_number=i,
                pretrained_weights=pretrained_weights,
                run_dir=run_dir,
            )
        
        # Step 4 with runtime thresholds
        if step_4_factory:
            tern_params = get_model_parameter_stats(pretrained_weights, steps[-1])
            print("\nTERNARIZATION PARAMETERS:")
            print(tern_params)
            
            step_4 = step_4_factory(tern_params)
            print(f"\nPERFORMING STEP {len(steps) + 1}/{len(steps) + 1}: {step_4.name}\n")
            
            pretrained_weights = train(
                step_config=step_4,
                step_number=len(steps) + 1,
                pretrained_weights=pretrained_weights,
                run_dir=run_dir,
            )
        
        # Save final config including step 4
        all_steps = list(steps) + [step_4]
        _save_experiment_config(run_dir, all_steps)
    else:
        _save_experiment_config(run_dir, steps)
    
    return pretrained_weights
```

---

## Updated `main.py`

```python
"""Entry point for MNIST OAR training."""

import tensorflow as tf

tf.random.set_seed(1997)

from experiments.mnist.config import perform_training_steps
from experiments.mnist.steps import STATIC_STEPS, create_step_4


def main():
    final_checkpoint = perform_training_steps(
        steps=STATIC_STEPS,
        step_4_factory=create_step_4,
    )
    print(f"\nTraining complete. Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
```

---

## Implementation Tasks

### Task 1: Create config dataclasses

**Files:**
- Create: `oar/config.py`
- Test: `tests/test_config_dataclasses.py`

**Step 1: Write failing tests**

```python
"""Tests for oar/config.py dataclasses."""

import pytest


class TestLayerConfig:
    def test_default_values(self):
        from oar.config import LayerConfig
        config = LayerConfig()
        assert config.activation == "tanh"
        assert config.gradient_scale == 1.0
        assert config.oar_lambda is None
        assert config.omega == 6
        assert config.quantize_threshold is None
    
    def test_validates_activation(self):
        from oar.config import LayerConfig
        with pytest.raises(ValueError, match="activation must be one of"):
            LayerConfig(activation="invalid")
    
    def test_validates_gradient_scale(self):
        from oar.config import LayerConfig
        with pytest.raises(ValueError, match="gradient_scale must be positive"):
            LayerConfig(gradient_scale=0)
    
    def test_validates_oar_lambda(self):
        from oar.config import LayerConfig
        with pytest.raises(ValueError, match="oar_lambda must be non-negative"):
            LayerConfig(oar_lambda=-1e-4)
    
    def test_allows_none_oar_lambda(self):
        from oar.config import LayerConfig
        config = LayerConfig(oar_lambda=None)
        assert config.oar_lambda is None


class TestInputConfig:
    def test_default_none(self):
        from oar.config import InputConfig
        config = InputConfig()
        assert config.quantize_threshold is None
    
    def test_validates_threshold(self):
        from oar.config import InputConfig
        with pytest.raises(ValueError, match="quantize_threshold must be non-negative"):
            InputConfig(quantize_threshold=-0.1)


class TestTrainingStepConfig:
    def test_requires_name(self):
        from oar.config import TrainingStepConfig
        config = TrainingStepConfig(name="test")
        assert config.name == "test"
        assert config.epochs == 100
        assert config.learning_rate == 1e-4
    
    def test_validates_epochs(self):
        from oar.config import TrainingStepConfig
        with pytest.raises(ValueError, match="epochs must be non-negative"):
            TrainingStepConfig(name="test", epochs=-1)
    
    def test_allows_zero_epochs(self):
        from oar.config import TrainingStepConfig
        config = TrainingStepConfig(name="test", epochs=0)
        assert config.epochs == 0


class TestResolveActivation:
    def test_tanh(self):
        from oar.config import resolve_activation
        import tensorflow as tf
        result = resolve_activation("tanh")
        assert result == tf.keras.activations.tanh
    
    def test_sign_ste_tanh(self):
        from oar.config import resolve_activation
        result = resolve_activation("sign_ste_tanh")
        assert callable(result)
    
    def test_mod_sign(self):
        from oar.config import resolve_activation
        result = resolve_activation("mod_sign", omega=6)
        assert callable(result)
    
    def test_invalid(self):
        from oar.config import resolve_activation
        with pytest.raises(ValueError, match="Unknown activation"):
            resolve_activation("invalid")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config_dataclasses.py -v
```

Expected: FAIL (module not found)

**Step 3: Implement dataclasses**

Create `oar/config.py` with the implementation shown above.

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config_dataclasses.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add oar/config.py tests/test_config_dataclasses.py
git commit -m "feat(oar): add config dataclasses for declarative step configuration"
```

---

### Task 2: Create MNIST steps

**Files:**
- Create: `experiments/mnist/steps.py`
- Test: `tests/test_mnist_steps.py`

**Step 1: Write failing tests**

```python
"""Tests for experiments/mnist/steps.py."""

import pytest


def test_static_steps_count():
    from experiments.mnist.steps import STATIC_STEPS
    assert len(STATIC_STEPS) == 3


def test_step_1_defaults():
    from experiments.mnist.steps import STEP_1
    assert STEP_1.name == "tanh_baseline"
    assert STEP_1.epochs == 100
    assert STEP_1.layers == {}


def test_step_2_gradient_scaling():
    from experiments.mnist.steps import STEP_2
    assert STEP_2.name == "sign_with_gradient_scaling"
    assert "QRNN_0" in STEP_2.layers
    assert STEP_2.layers["QRNN_0"].gradient_scale == 4.0
    assert STEP_2.layers["QRNN_0"].activation == "sign_ste_tanh"


def test_step_3_input_quantization():
    from experiments.mnist.steps import STEP_3
    assert STEP_3.input_config.quantize_threshold == 0.7


def test_create_step_4():
    from experiments.mnist.steps import create_step_4
    
    tern_params = {"QRNN_0": 0.1, "QRNN_1": 0.1, "DENSE_0": 0.1, "DENSE_OUT": 0.1}
    step = create_step_4(tern_params, t=1.5)
    
    assert step.name == "full_quantization_with_oar"
    assert step.learning_rate == 1e-5
    assert step.layers["QRNN_0"].quantize_threshold == 0.15  # 1.5 * 0.1
    assert step.layers["QRNN_0"].oar_lambda == 1e-4


def test_get_default_layer_config():
    from experiments.mnist.steps import get_default_layer_config
    
    # Regular layer defaults to tanh
    cfg = get_default_layer_config("QRNN_0")
    assert cfg.activation == "tanh"
    
    # DENSE_OUT defaults to softmax
    cfg = get_default_layer_config("DENSE_OUT")
    assert cfg.activation == "softmax"
```

**Step 2: Run tests**

```bash
pytest tests/test_mnist_steps.py -v
```

**Step 3: Implement**

Create `experiments/mnist/steps.py` with the implementation shown above.

**Step 4: Run tests**

```bash
pytest tests/test_mnist_steps.py -v
```

**Step 5: Commit**

```bash
git add experiments/mnist/steps.py tests/test_mnist_steps.py
git commit -m "feat(mnist): add declarative step definitions"
```

---

### Task 3: Update get_model() signature

**Files:**
- Modify: `experiments/mnist/model.py`
- Test: `tests/test_mnist_model.py`

**Step 1: Write failing test**

```python
def test_get_model_accepts_step_config():
    from experiments.mnist.steps import STEP_1
    from experiments.mnist.model import get_model
    
    model = get_model(STEP_1)
    assert model is not None
    assert "MNIST_RNN" in model.name
```

**Step 2-5:** Update `get_model()` to accept `TrainingStepConfig`, run tests, commit.

---

### Task 4: Update train() signature

**Files:**
- Modify: `experiments/mnist/training.py`
- Modify: `experiments/mnist/config.py` (update orchestration)
- Test: `tests/test_mnist_training.py`

**Step 1: Write failing test**

```python
def test_train_accepts_step_config():
    from experiments.mnist.steps import STEP_1
    from experiments.mnist.training import train
    import inspect
    
    sig = inspect.signature(train)
    params = list(sig.parameters.keys())
    assert "step_config" in params
    assert "options" not in params
    assert "layer_options" not in params
```

**Step 2-5:** Update `train()` signature, update `_save_step_config()`, run tests, commit.

---

### Task 5: Update orchestration

**Files:**
- Modify: `experiments/mnist/config.py`
- Modify: `experiments/mnist/__init__.py`

**Implementation:**
- Delete `perform_step_in_four_step_quant()`
- Delete `perform_four_step_quant()`
- Delete `get_default_layer_options()`
- Delete `_make_oar_config()`
- Add `perform_training_steps()`
- Update `__init__.py` exports

---

### Task 6: Update main.py

**Files:**
- Modify: `main.py`

**Implementation:**

```python
from experiments.mnist.config import perform_training_steps
from experiments.mnist.steps import STATIC_STEPS, create_step_4

final_checkpoint = perform_training_steps(
    steps=STATIC_STEPS,
    step_4_factory=create_step_4,
)
```

---

### Task 7: Export from oar package

**Files:**
- Modify: `oar/__init__.py`

**Implementation:**

```python
from oar.config import (
    LayerConfig,
    InputConfig,
    TrainingStepConfig,
    resolve_activation,
)

__all__ = [
    # ... existing ...
    "LayerConfig",
    "InputConfig",
    "TrainingStepConfig",
    "resolve_activation",
]
```

---

### Task 8: Delete old code

**Files:**
- Modify: `experiments/mnist/config.py`

**Delete:**
- `MNIST_OPTIONS` dict (replaced by `TrainingStepConfig`)
- `ENLARGED_MNIST_OPTIONS` dict
- `_make_oar_config()` helper
- `get_default_layer_options()` function
- `perform_step_in_four_step_quant()` function
- `perform_four_step_quant()` function

---

### Task 9: Update tests

**Files:**
- Modify: `tests/test_mnist_config.py`

**Changes:**
- Remove tests for deleted functions
- Update imports
- Add tests for new `perform_training_steps()`

---

### Task 10: Integration test

**Validation:**
1. `pytest tests/ -v` - all tests pass
2. Run smoke test: 1 epoch per step
3. Verify `config.json` contains step configs
4. Verify model trains correctly

```bash
# Quick smoke test
python -c "
from experiments.mnist.config import perform_training_steps
from experiments.mnist.steps import STATIC_STEPS, create_step_4
from oar.config import TrainingStepConfig

# Create minimal test steps
test_steps = [TrainingStepConfig(name='test', epochs=1)]
perform_training_steps(test_steps, step_4_factory=None)
"
```

---

## Summary

**What Changed from Original Plan:**

| Original | Simplified |
|----------|------------|
| `OARConfig` nested dataclass | Inline `oar_lambda: float \| None` |
| `TernaryQuantizationConfig` nested | Inline `quantize_threshold: float \| None` |
| `build_layer_options()` conversion | Deleted - direct consumption |
| `options` + `layer_options` dicts | Single `TrainingStepConfig` |
| `get_model(options, layer_options)` | `get_model(step_config)` |
| `train(options, layer_options)` | `train(step_config)` |
| Hardcoded layer names in oar/ | Layer names in experiments/mnist/ |

**Benefits:**
- No conversion layer
- Single source of truth (`TrainingStepConfig`)
- Simpler, flatter dataclasses
- MNIST-specific logic in MNIST folder
- Easy to add Penn Treebank (define `PTB_STEPS` with different layers)
- Less code overall
