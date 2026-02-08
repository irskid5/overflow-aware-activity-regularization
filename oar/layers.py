"""OAR layer implementations for quantized RNNs and Dense layers."""

import tensorflow as tf
import keras.backend as K
from qkeras import QSimpleRNNCell, QDense

from oar.regularizers import OAR2
from oar.callbacks import _reservoir_update, RESERVOIR_UPDATE_EVERY


@tf.keras.utils.register_keras_serializable(package="OAR")
class Downsampling(tf.keras.layers.Layer):
    """
    Temporal downsampling layer for RNN outputs.
    
    Reduces sequence length by a factor while increasing feature dimension.
    Used between RNN layers to reduce computation.
    
    Args:
        reduction_factor: Factor by which to reduce sequence length
        batch_size: Batch size (optional, for static shape inference)
    """

    def __init__(self, reduction_factor, batch_size=None, **kwargs):
        super(Downsampling, self).__init__(**kwargs)
        self.reduction_factor = reduction_factor
        self.batch_size = batch_size

    def compute_output_shape(self, input_shape):
        max_time = input_shape[1]
        num_units = input_shape[2]
        if max_time is not None:
            extra_timestep = tf.math.floormod(max_time, self.reduction_factor)
            reduced_size = (
                tf.math.floordiv(max_time, self.reduction_factor) + extra_timestep
            )
        else:
            reduced_size = None
        return [input_shape[0], reduced_size, num_units * self.reduction_factor]

    def call(self, inputs):
        input_shape = K.int_shape(inputs)

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = input_shape[0]

        outputs = inputs

        if input_shape[1] is not None:
            max_time = input_shape[1]
            extra_timestep = tf.math.floormod(max_time, self.reduction_factor)

            paddings = [[0, 0], [0, extra_timestep], [0, 0]]
            outputs = tf.pad(outputs, paddings)

        else:
            outputs = tf.signal.frame(
                outputs,
                self.reduction_factor,
                self.reduction_factor,
                pad_end=False,
                axis=1,
            )

        out_shape = self.compute_output_shape(input_shape)
        out_shape_tuple = tuple(-1 if s is None else s for s in out_shape)

        return tf.reshape(outputs, out_shape_tuple)

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_factor": self.reduction_factor,
            "batch_size": self.batch_size,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="OAR")
class QSimpleRNNCellWithOAR(QSimpleRNNCell):
    """
    QKeras SimpleRNN cell with Overflow-Aware Activity Regularization.
    
    Extends QSimpleRNNCell to apply OAR regularization on pre-activations
    and maintain reservoir sampling for distribution visualization.
    
    Args:
        units: Number of hidden units
        batch_size: Batch size for static shape inference
        activation: Activation function
        use_bias: Whether to use bias
        kernel_quantizer: Quantizer for input weights
        recurrent_quantizer: Quantizer for recurrent weights
        use_oar: Enable OAR regularization
        oar_lambda: OAR regularization rate
        omega: Bit precision (k = 2^omega)
        s: Gradient scaling factor
        reservoir_size: Size of pre-activation reservoir
    """

    def __init__(
        self,
        units,
        batch_size=512,
        activation=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        kernel_quantizer=None,
        recurrent_quantizer=None,
        bias_quantizer=None,
        state_quantizer=None,
        use_oar=False,
        oar_lambda=0,
        omega=32,
        s=1,
        reservoir_size=100000,
        **kwargs
    ):
        super(QSimpleRNNCellWithOAR, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            kernel_quantizer=kernel_quantizer,
            recurrent_quantizer=recurrent_quantizer,
            bias_quantizer=bias_quantizer,
            state_quantizer=state_quantizer,
            **kwargs
        )

        # OAR regularization
        self.oar = None
        self.oar_lambda = oar_lambda
        self.omega = omega
        if use_oar:
            self.oar = OAR2(oar_lambda=oar_lambda, k=2**omega, name=kwargs.get("name", ""))

        # Gradient scaling
        self.s = s

        # Batch size and reservoir
        self.batch_size = batch_size
        self.reservoir_size = reservoir_size

    def build(self, input_shape):
        super(QSimpleRNNCellWithOAR, self).build(input_shape)

        # Debug weights for quantized kernels
        if self.kernel_quantizer:
            self.quantized_kernel = self.add_weight(
                name="quantized_kernel",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=False,
            )
        if self.recurrent_quantizer:
            self.quantized_recurrent_kernel = self.add_weight(
                name="quantized_recurrent_kernel",
                shape=self.recurrent_kernel.shape,
                dtype=self.recurrent_kernel.dtype,
                initializer="zeros",
                trainable=False,
            )

        # EMA statistics
        self.wx = self.add_weight(
            name="wx_abs",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )
        self.wh = self.add_weight(
            name="wh_abs",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )

        # Reservoir sampling for pre-activations
        self.preact_reservoir = self.add_weight(
            name="preact_reservoir",
            shape=[self.reservoir_size],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )
        self.reservoir_count = self.add_weight(
            name="reservoir_count",
            shape=[],
            dtype=tf.int64,
            initializer="zeros",
            trainable=False,
        )
        self.reservoir_step = self.add_weight(
            name="reservoir_step",
            shape=[],
            dtype=tf.int64,
            initializer="zeros",
            trainable=False,
        )

    def call(self, inputs, states, training=None):
        prev_output = states[0] if tf.nest.is_nested(states) else states

        # Quantize state
        if self.state_quantizer:
            quantized_prev_output = self.state_quantizer_internal(prev_output)
        else:
            quantized_prev_output = prev_output

        # Quantize kernel
        if self.kernel_quantizer:
            quantized_kernel = self.kernel_quantizer_internal(self.kernel)
            self.quantized_kernel.assign(quantized_kernel)
        else:
            quantized_kernel = self.kernel

        h = K.dot(inputs, quantized_kernel)

        # Quantize recurrent kernel
        if self.recurrent_quantizer:
            quantized_recurrent = self.recurrent_quantizer_internal(self.recurrent_kernel)
            self.quantized_recurrent_kernel.assign(quantized_recurrent)
        else:
            quantized_recurrent = self.recurrent_kernel

        h_2 = K.dot(quantized_prev_output, quantized_recurrent)

        # Update EMA stats
        self.wx.assign(0.90 * self.wx + 0.10 * tf.reduce_mean(tf.abs(h), axis=0))
        self.wh.assign(0.90 * self.wh + 0.10 * tf.reduce_mean(tf.abs(h_2), axis=0))

        # Gradient scaling (forward unscaled, backward scaled)
        s = self.s
        output = h / s + h_2 / s + tf.stop_gradient(-h / s - h_2 / s + h + h_2)

        # Reservoir sampling for pre-activation histogram
        self.reservoir_step.assign_add(1)
        do_update = tf.equal(
            tf.math.floormod(self.reservoir_step, RESERVOIR_UPDATE_EVERY), 0
        )
        updated_reservoir, updated_count = tf.cond(
            do_update,
            lambda: _reservoir_update(
                reservoir=self.preact_reservoir,
                count=tf.cast(self.reservoir_count, tf.int64),
                new_values=h + h_2,
                reservoir_size=self.reservoir_size,
                seed=1997,
            ),
            lambda: (self.preact_reservoir, self.reservoir_count),
        )
        self.preact_reservoir.assign(updated_reservoir)
        self.reservoir_count.assign(updated_count)

        # Apply OAR and activation
        if self.activation is not None:
            if self.oar is not None:
                output = self.oar(output)
            output = self.activation(output)

        return output, [output]

    def get_config(self):
        config = super().get_config()
        config.update({
            "batch_size": self.batch_size,
            "use_oar": self.oar is not None,
            "oar_lambda": self.oar_lambda,
            "omega": self.omega,
            "s": self.s,
            "reservoir_size": self.reservoir_size,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="OAR")
class QRNNWithOAR(tf.keras.layers.RNN):
    """
    RNN layer wrapper for QSimpleRNNCellWithOAR.
    
    Provides a convenient interface matching Keras RNN API while
    internally using QSimpleRNNCellWithOAR for OAR support.
    
    Forces unroll=True when OAR is enabled (required for per-timestep add_loss).
    """

    def __init__(
        self,
        cell=None,
        units=128,
        activation="tanh",
        batch_size=512,
        stateful=False,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_quantizer=None,
        recurrent_quantizer=None,
        bias_quantizer=None,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer=None,
        use_bias=False,
        use_oar=False,
        oar_lambda=0,
        omega=32,
        s=1.0,
        reservoir_size=100000,
        unroll=False,
        name="",
        **kwargs
    ):
        # Store params
        self.use_oar = use_oar
        self.oar_lambda = oar_lambda
        self.omega = omega
        self.s = s
        self.reservoir_size = reservoir_size
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer

        # Force unroll when OAR enabled
        to_unroll = unroll or use_oar

        cell = (
            QSimpleRNNCellWithOAR(
                units,
                activation=activation,
                batch_size=batch_size,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_quantizer=kernel_quantizer,
                recurrent_quantizer=recurrent_quantizer,
                bias_quantizer=bias_quantizer,
                use_bias=use_bias,
                use_oar=use_oar,
                oar_lambda=oar_lambda,
                omega=omega,
                s=s,
                reservoir_size=reservoir_size,
                name=name,
            )
            if cell is None
            else cell
        )

        super(QRNNWithOAR, self).__init__(
            cell, return_sequences=True, stateful=stateful, unroll=to_unroll, name=name
        )

        self.quantizers = self.get_quantizers()

    # Properties to forward from self.cell
    _CELL_FORWARDED_ATTRS = frozenset({
        "units", "activation", "use_bias", "bias_initializer",
        "kernel_regularizer", "recurrent_regularizer", "bias_regularizer",
        "kernel_constraint", "recurrent_constraint", "bias_constraint",
        "kernel_quantizer_internal", "recurrent_quantizer_internal",
        "bias_quantizer_internal", "state_quantizer_internal",
        "kernel_quantizer", "recurrent_quantizer", "bias_quantizer", "state_quantizer",
    })

    def get_quantizers(self):
        return self.cell.quantizers

    def get_prunable_weights(self):
        return [self.cell.kernel, self.cell.recurrent_kernel]

    def __getattr__(self, name):
        """Delegate attribute access to self.cell for forwarded properties."""
        if name in QRNNWithOAR._CELL_FORWARDED_ATTRS:
            return getattr(self.cell, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_config(self):
        from tensorflow.keras import constraints
        config = super().get_config()
        config.update({
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "kernel_regularizer": constraints.serialize(self.kernel_regularizer),
            "recurrent_regularizer": constraints.serialize(self.recurrent_regularizer),
            "bias_regularizer": constraints.serialize(self.bias_regularizer),
            "kernel_quantizer": constraints.serialize(self.kernel_quantizer_internal),
            "recurrent_quantizer": constraints.serialize(self.recurrent_quantizer_internal),
            "bias_quantizer": constraints.serialize(self.bias_quantizer_internal),
        })
        return config


@tf.keras.utils.register_keras_serializable(package="OAR")
class QDenseWithOAR(QDense):
    """
    Quantized Dense layer with OAR support.
    
    Extends QDense with overflow-aware regularization and
    reservoir sampling for pre-activation distribution tracking.
    """

    def __init__(
        self,
        units,
        activation=None,
        batch_size=512,
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        kernel_quantizer=None,
        bias_quantizer=None,
        kernel_range=None,
        bias_range=None,
        use_oar=False,
        oar_lambda=0,
        omega=32,
        s=1,
        reservoir_size=100000,
        **kwargs
    ):
        # OAR setup
        self.oar = None
        self.oar_lambda = oar_lambda
        self.omega = omega
        if use_oar:
            self.oar = OAR2(oar_lambda=oar_lambda, k=2**omega, name=kwargs.get("name", ""))

        self.s = s
        self.batch_size = batch_size
        self.reservoir_size = reservoir_size

        super(QDenseWithOAR, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kernel_quantizer=kernel_quantizer,
            bias_quantizer=bias_quantizer,
            kernel_range=kernel_range,
            bias_range=bias_range,
            **kwargs
        )

    def build(self, input_shape):
        super(QDenseWithOAR, self).build(input_shape)

        if self.kernel_quantizer:
            self.quantized_kernel = self.add_weight(
                name="quantized_kernel",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=False,
            )

        self.wx = self.add_weight(
            name="wx_abs",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )

        self.preact_reservoir = self.add_weight(
            name="preact_reservoir",
            shape=[self.reservoir_size],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )
        self.reservoir_count = self.add_weight(
            name="reservoir_count",
            shape=[],
            dtype=tf.int64,
            initializer="zeros",
            trainable=False,
        )
        self.reservoir_step = self.add_weight(
            name="reservoir_step",
            shape=[],
            dtype=tf.int64,
            initializer="zeros",
            trainable=False,
        )

        self.x_input = self.add_weight(
            name="x_abs",
            shape=[input_shape[-1]],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False,
        )

    def call(self, inputs):
        # Input stats
        reduce_dims = tf.range(0, tf.rank(inputs) - 1)
        self.x_input.assign(
            0.9 * self.x_input + 0.1 * tf.reduce_mean(tf.abs(inputs), axis=reduce_dims)
        )

        # Quantize kernel
        quantized_kernel = self.kernel
        if self.kernel_quantizer:
            quantized_kernel = self.kernel_quantizer_internal(self.kernel)
            self.quantized_kernel.assign(quantized_kernel)

        # Forward pass
        h = tf.keras.backend.dot(inputs, quantized_kernel)

        # Gradient scaling
        s = self.s
        h = h / s + tf.stop_gradient(-h / s + h)

        # Update stats
        reduce_dims = tf.range(0, tf.rank(h) - 1)
        self.wx.assign(
            0.9 * self.wx + 0.1 * tf.reduce_mean(tf.abs(h), axis=reduce_dims)
        )

        # Reservoir sampling
        self.reservoir_step.assign_add(1)
        do_update = tf.equal(
            tf.math.floormod(self.reservoir_step, RESERVOIR_UPDATE_EVERY), 0
        )
        updated_reservoir, updated_count = tf.cond(
            do_update,
            lambda: _reservoir_update(
                reservoir=self.preact_reservoir,
                count=tf.cast(self.reservoir_count, tf.int64),
                new_values=h,
                reservoir_size=self.reservoir_size,
                seed=1997,
            ),
            lambda: (self.preact_reservoir, self.reservoir_count),
        )
        self.preact_reservoir.assign(updated_reservoir)
        self.reservoir_count.assign(updated_count)

        output = h
        if self.activation is not None:
            if self.oar is not None:
                output = self.oar(output)
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "batch_size": self.batch_size,
            "use_oar": self.oar is not None,
            "oar_lambda": self.oar_lambda,
            "omega": self.omega,
            "s": self.s,
            "reservoir_size": self.reservoir_size,
        })
        return config
