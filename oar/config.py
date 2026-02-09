"""Configuration dataclasses for OAR training."""

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
