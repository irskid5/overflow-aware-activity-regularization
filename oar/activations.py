"""OAR activation functions and wrappers."""

import tensorflow as tf


def sign_ste_tanh(x):
    """
    Sign activation with tanh gradient (straight-through estimator).
    
    Forward: sign(x)
    Backward: tanh'(x) = 1 - tanh(x)^2
    
    Handles x=0 by returning +1 (adds 1.0 - |sign(x)| to fill zeros).
    
    Args:
        x: Input tensor
    
    Returns:
        Sign of x with tanh gradient for backprop
    """
    out = tf.keras.activations.tanh(x)
    q = tf.math.sign(x)
    q += 1.0 - tf.math.abs(q)  # Handle x=0 -> +1
    return out + tf.stop_gradient(-out + q)


def mod_sign(x, num_bits=8):
    """
    Modular sign activation (ModSign from paper, Eq. 4).
    
    Computes sign(x mod 2^num_bits) with tanh gradient.
    Used in Step 4 of four-step quantization for TFHE compatibility.
    
    Args:
        x: Input tensor (pre-activations)
        num_bits: Bit precision (omega), determines modulus 2^num_bits
    
    Returns:
        Signed modular result with tanh gradient for backprop
    """
    def _inner_fn(x, num_bits):
        base = 2**num_bits
        half_base = 2 ** (num_bits - 1)

        # Cast to int for modular reduction
        x_int = tf.cast(x, tf.int32)

        # Modular reduction (unsigned)
        modded = tf.math.mod(x_int, base)

        # Convert to signed representation
        signed = tf.where(tf.greater_equal(modded, half_base), modded - base, modded)

        return tf.cast(signed, tf.float32)

    # Regular sign with STE, then replace forward value with modular version
    out = sign_ste_tanh(x) + tf.stop_gradient(
        -sign_ste_tanh(x) + sign_ste_tanh(_inner_fn(x, num_bits=num_bits))
    )

    return out


@tf.keras.utils.register_keras_serializable(package="OAR")
class TrackedActivation(tf.keras.layers.Layer):
    """
    Activation wrapper that tracks input/output statistics.
    
    Maintains exponential moving averages of mean and std for both
    input and output of the activation function. Useful for monitoring
    distribution shifts during training.
    
    Args:
        activation: Activation function (callable or string)
        name: Layer name
    """

    def __init__(self, activation=None, name="", **kwargs):
        super(TrackedActivation, self).__init__(name=name, **kwargs)
        self._activation = activation
        # Resolve string activations to callables
        if isinstance(activation, str):
            self._activation_fn = tf.keras.activations.get(activation)
        else:
            self._activation_fn = activation

    def build(self, input_shape):
        self.inp_moving_mean = self.add_weight(
            name="inp_moving_mean",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        self.inp_moving_std = self.add_weight(
            name="inp_moving_std",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        self.out_moving_mean = self.add_weight(
            name="out_moving_mean",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        self.out_moving_std = self.add_weight(
            name="out_moving_std",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        super(TrackedActivation, self).build(input_shape)

    def call(self, inputs):
        out = inputs

        # Apply activation
        if self._activation_fn:
            out = self._activation_fn(out)

        # Update statistics
        self.inp_moving_mean.assign(
            0.9 * self.inp_moving_mean + 0.1 * tf.reduce_mean(inputs)
        )
        self.inp_moving_std.assign(
            0.9 * self.inp_moving_std + 0.1 * tf.math.reduce_std(inputs)
        )
        self.out_moving_mean.assign(
            0.9 * self.out_moving_mean + 0.1 * tf.reduce_mean(out)
        )
        self.out_moving_std.assign(
            0.9 * self.out_moving_std + 0.1 * tf.math.reduce_std(out)
        )

        return out

    def get_config(self):
        config = super().get_config()
        # Serialize activation - handle both callables and strings
        if isinstance(self._activation, str):
            activation_config = self._activation
        elif self._activation is None:
            activation_config = None
        else:
            # Try to serialize as Keras activation
            try:
                activation_config = tf.keras.activations.serialize(self._activation)
            except (TypeError, ValueError):
                # For custom functions, store as None and warn
                activation_config = None
        config.update({
            "activation": activation_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize activation if needed
        activation = config.pop("activation", None)
        if isinstance(activation, dict):
            activation = tf.keras.activations.deserialize(activation)
        elif isinstance(activation, str):
            activation = tf.keras.activations.get(activation)
        return cls(activation=activation, **config)
