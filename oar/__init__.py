"""
Overflow-Aware Activity Regularization (OAR) Package.

A TensorFlow/Keras framework for training quantized neural networks
with overflow-aware regularization for TFHE inference.
"""

from oar.regularizers import OAR1, OAR2, OARRegularizer, oar_penalty_fn, compute_oar_metric
from oar.activations import sign_ste_tanh, mod_sign, TrackedActivation, resolve_activation
from oar.quantizers import TernarizationWithThreshold, ternarize_tensor_with_threshold
from oar.callbacks import (
    ReservoirHistogramCallback,
    ResetStatesCallback,
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
from oar.metrics import Perplexity
from oar.runner import Runner, ModelFactory, DataLoader, tee_output
from oar.config import (
    ActivationConfig,
    OARConfig,
    QuantizationConfig,
    LayerStepConfig,
    StepConfig,
    ExperimentConfig,
    DACITE_CONFIG,
)
from oar.model_utils import compute_thresholds_if_needed

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
    "resolve_activation",
    # Quantizers
    "TernarizationWithThreshold",
    "ternarize_tensor_with_threshold",
    # Callbacks
    "ReservoirHistogramCallback",
    "ResetStatesCallback",
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
    # Metrics
    "Perplexity",
    # Runner
    "Runner",
    "ModelFactory",
    "DataLoader",
    "tee_output",
    # Config
    "ActivationConfig",
    "OARConfig",
    "QuantizationConfig",
    "LayerStepConfig",
    "StepConfig",
    "ExperimentConfig",
    "DACITE_CONFIG",
    # Model utils
    "compute_thresholds_if_needed",
]
