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
