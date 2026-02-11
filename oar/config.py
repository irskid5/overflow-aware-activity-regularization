"""Configuration dataclasses for OAR training.

Provides a hierarchical configuration structure for experiments:
- ActivationConfig: activation function settings
- OARConfig: overflow-aware activity regularization settings
- QuantizationConfig: weight/input ternarization settings
- LayerStepConfig: per-layer configuration at one training step
- StepConfig: configuration for one training step
- ExperimentConfig: full experiment configuration

Uses dacite for dict→dataclass conversion with DACITE_CONFIG.
"""

from dataclasses import dataclass, field
from typing import Literal

from dacite import Config


@dataclass
class ActivationConfig:
    """Activation function settings.
    
    Attributes:
        function: Activation function name
        gradient_scale: s - gradient scaling factor (for STE activations)
        omega: Bit precision (required for mod_sign)
    """
    function: Literal["tanh", "sign_ste_tanh", "mod_sign", "softmax"] = "tanh"
    gradient_scale: float | None = None
    omega: int | None = None  # Required for mod_sign
    
    def __post_init__(self):
        valid = {"tanh", "sign_ste_tanh", "mod_sign", "softmax"}
        if self.function not in valid:
            raise ValueError(f"function must be one of {valid}")
        if self.function == "mod_sign" and self.omega is None:
            raise ValueError("omega required for mod_sign")


@dataclass
class OARConfig:
    """OAR (overflow-aware activity regularization) settings.
    
    Attributes:
        regularization_rate: λ - OAR regularization rate (0 = observe only)
        omega: Bit precision (k = 2^omega modulus)
    """
    regularization_rate: float = 0.0  # 0 = observe only
    omega: int = 6
    
    def __post_init__(self):
        if self.regularization_rate < 0:
            raise ValueError("regularization_rate must be non-negative")
        if self.omega < 1:
            raise ValueError("omega must be >= 1")


@dataclass
class QuantizationConfig:
    """Quantization settings for a layer.
    
    For INPUT layer: threshold is fixed (ternarization_scale used directly as threshold)
    For other layers: threshold = ternarization_scale × E[|θ|] (computed by model_factory)
    
    Attributes:
        ternarization_scale: t - threshold scale factor (threshold = t × E[|values|])
        threshold: Direct threshold value (overrides ternarization_scale computation)
        oar: OAR regularization settings (None = disabled)
    """
    ternarization_scale: float | None = None  # t: threshold = t × E[|values|]
    threshold: float | None = None  # Override (skip computation)
    oar: OARConfig | None = None
    
    def __post_init__(self):
        if self.ternarization_scale is not None and self.ternarization_scale < 0:
            raise ValueError("ternarization_scale must be non-negative")
        if self.threshold is not None and self.threshold < 0:
            raise ValueError("threshold must be non-negative")


@dataclass
class LayerStepConfig:
    """Configuration for one layer at one training step.
    
    Attributes:
        activation: Activation function configuration
        quantization: Quantization/ternarization configuration
    """
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


@dataclass
class StepConfig:
    """Configuration for one training step.
    
    Attributes:
        name: Human-readable step name (for logging)
        epochs: Training epochs
        learning_rate: Learning rate
        cosine_decay_epochs: Number of epochs over which to decay LR (epoch-based, not step-based)
        cosine_decay_alpha: Minimum LR ratio (0.0 = decay to 0, 0.1 = floor at 10% of initial)
        batch_size: Batch size
        enlarge: Use enlarged input (128x128) vs standard (28x28) for MNIST
        layers: Per-layer configuration dict (keys are layer names like "INPUT", "QRNN_0", etc.)
    """
    name: str
    epochs: int = 1000
    learning_rate: float = 1e-4
    cosine_decay_epochs: int = 100  # Decay LR over this many epochs (matches reference)
    cosine_decay_alpha: float = 0.1  # LR floors at alpha * initial_lr (0.1 = 10%)
    batch_size: int = 512
    enlarge: bool = False  # 28x28 (False) or 128x128 (True) for MNIST
    layers: dict[str, LayerStepConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 <= self.cosine_decay_alpha <= 1.0:
            raise ValueError("cosine_decay_alpha must be between 0.0 and 1.0")
        if self.cosine_decay_epochs < 1:
            raise ValueError("cosine_decay_epochs must be >= 1")


@dataclass
class ExperimentConfig:
    """Full experiment configuration.
    
    Attributes:
        name: Experiment name (used for run directory naming)
        layer_names: Ordered list of layer names in the model
        runs_dir: Base directory for experiment outputs
        steps: Training steps indexed by step number
    """
    name: str
    layer_names: list[str]
    runs_dir: str = "runs/"
    steps: dict[int, StepConfig] = field(default_factory=dict)


# String→ActivationConfig hook for shorthand like "tanh" instead of {"function": "tanh"}
def _activation_hook(data: dict | str) -> dict:
    if isinstance(data, str):
        return {"function": data}
    return data


DACITE_CONFIG = Config(
    cast=[Literal],
    type_hooks={ActivationConfig: _activation_hook},
)
