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
