"""OAR regularizers for overflow-aware training."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="OAR")
def oar_penalty_fn(x, k, a):
    """
    OAR penalty function (Equation 1 from paper).
    
    OAR₁(x, k) = ReLU(1 - (4/k) · | (|x + 0.5| - k/4) mod k - k/2 |)
    
    Args:
        x: Input tensor (pre-activations)
        k: Modulus size (2^omega)
        a: Amplitude scaling factor
    
    Returns:
        Penalty tensor (same shape as x)
    """
    inner = tf.abs(x + 0.5) - k / 4
    modded = tf.math.mod(inner, k)
    out = 1 - (4 / k) * tf.abs(modded - k / 2)
    return a * tf.nn.relu(out)


def compute_oar_metric(x, k, a):
    """
    Compute fraction of pre-activations in valid modular range.
    
    Args:
        x: Input tensor (pre-activations)
        k: Modulus size (2^omega)
        a: Amplitude scaling factor
    
    Returns:
        Fraction of values NOT in overflow region (higher is better)
    """
    wrongs = tf.sign(oar_penalty_fn(x, k=k, a=a))
    rights_ratio = 1 - tf.reduce_mean(wrongs, axis=[-1])
    return rights_ratio


@tf.keras.utils.register_keras_serializable(package="OAR")
class OARRegularizer(tf.keras.layers.Layer):
    """
    Base Overflow-Aware Activity Regularizer.
    
    Applies a penalty for pre-activations in overflow regions.
    
    Args:
        oar_lambda: Regularization rate (lambda in paper)
        k: Modulus size, typically 2^omega where omega is bit-width
        a: Amplitude scaling factor (default 1.0)
        squared: If True, square the penalty (OAR2); if False, linear (OAR1)
        name: Layer name for metric tracking
    """

    def __init__(self, oar_lambda=1e-3, k=2**8, a=1.0, squared=False, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.oar_lambda = oar_lambda
        self.k = k
        self.a = a
        self.squared = squared
        metric_prefix = "OAR2" if squared else "OAR1"
        self.no_acc_metric = tf.keras.metrics.Mean(name=f"{metric_prefix}/{name}")

    def call(self, x):
        penalty = oar_penalty_fn(x=x, k=self.k, a=self.a)
        if self.squared:
            penalty = tf.square(penalty)
        
        loss = self.oar_lambda * tf.reduce_sum(penalty)
        accuracy = compute_oar_metric(x, k=self.k, a=self.a)
        
        self.add_loss(loss)
        self.add_metric(self.no_acc_metric(accuracy))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "oar_lambda": float(self.oar_lambda),
            "k": int(self.k),
            "a": float(self.a),
            "squared": bool(self.squared),
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Backward-compatible factory functions
def OAR1(oar_lambda=1e-3, k=2**8, a=1.0, name="", **kwargs):
    """OAR₁ - linear penalty. Returns OARRegularizer with squared=False."""
    return OARRegularizer(oar_lambda=oar_lambda, k=k, a=a, squared=False, name=name, **kwargs)


def OAR2(oar_lambda=1e-3, k=2**8, a=1.0, name="", **kwargs):
    """OAR₂ - squared penalty. Returns OARRegularizer with squared=True."""
    return OARRegularizer(oar_lambda=oar_lambda, k=k, a=a, squared=True, name=name, **kwargs)


# Add from_config to factory functions for test compatibility
# Filter config to only include OAR1/OAR2 parameters (exclude squared and base Layer params)
def _oar1_from_config(config):
    filtered = {k: v for k, v in config.items() if k in ('oar_lambda', 'k', 'a', 'name')}
    return OAR1(**filtered)

def _oar2_from_config(config):
    filtered = {k: v for k, v in config.items() if k in ('oar_lambda', 'k', 'a', 'name')}
    return OAR2(**filtered)

OAR1.from_config = _oar1_from_config
OAR2.from_config = _oar2_from_config
