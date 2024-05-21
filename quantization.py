import tensorflow as tf
import keras.backend as K
from qkeras import *

from tensorflow_model_optimization.python.core.quantization.keras.quantizers import *


def ternarize_tensor_with_threshold(x, theta=1.0):
    """
    Ternary quantizer where
    x = -1 if x-mean <= -threshold,
    x = 0  if -threshold < x-mean < threshold,
    x = 1  if x-mean >= threshold.

    """
    q = K.cast(tf.abs(x) >= theta, K.floatx()) * tf.sign(x)
    return q


class TernarizationWithThreshold(BaseQuantizer):
    """Performs ternarization with a threshold.

    Args:
        BaseQuantizer (_type_): _description_
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
        """Get the maximum value that ternary can respresent."""
        return 1.0

    def min(self):
        """Get the minimum value that ternary can respresent."""
        return -1.0

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "threshold": self.threshold,
        }
        return config
