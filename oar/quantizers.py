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
