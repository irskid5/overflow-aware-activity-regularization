"""
Backward compatibility shim for quantization.py.

Quantizers have been moved to the oar/ package.
This module re-exports them for backward compatibility.

DEPRECATED: Import from oar package directly:
    from oar import TernarizationWithThreshold, ternarize_tensor_with_threshold
"""

from oar import TernarizationWithThreshold, ternarize_tensor_with_threshold

__all__ = ["TernarizationWithThreshold", "ternarize_tensor_with_threshold"]
