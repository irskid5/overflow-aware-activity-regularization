"""
Overflow-Aware Activity Regularization (OAR) Package.

A TensorFlow/Keras framework for training quantized neural networks
with overflow-aware regularization for TFHE inference.
"""

from oar.regularizers import OAR1, OAR2, OARRegularizer, oar_penalty_fn, compute_oar_metric
from oar.activations import sign_ste_tanh, mod_sign, TrackedActivation

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
]
