from math import pi
import tensorflow as tf
import keras.backend as K
from keras.engine import data_adapter
import numpy as np
# GENERAL UTILS
from utils.general import *

# QUANTIZATION FUNCTIONS
from quantization import *

from qkeras import *


class QRNNWithOAR(tf.keras.layers.RNN):
    def __init__(
            self,
            cell=None,
            units=256,
            activation="tanh",
            stateful=False,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            kernel_quantizer=None,
            recurrent_quantizer=None,
            bias_quantizer=None,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer=None,
            use_bias=False,
            use_oar=False,
            oar_lm=0,
            oar_bits=32,
            s=1.0,
            unroll=False,
            name="", **kwargs):

        # Initializers
        k_init = kernel_initializer  # 'glorot_uniform'
        rk_init = recurrent_initializer  # 'orthogonal'

        # Overflow-Aware Activity Regularization
        self.use_oar = use_oar
        self.oar_lm = oar_lm
        self.oar_bits = oar_bits

        # Gradient scaling
        self.s = s

        # These are flags that require rnn unrolling
        to_unroll = unroll or use_oar

        cell = QSimpleRNNCellWithOAR(
            units,
            activation=activation,
            kernel_initializer=k_init,
            recurrent_initializer=rk_init,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_quantizer=kernel_quantizer,
            recurrent_quantizer=recurrent_quantizer,
            bias_quantizer=bias_quantizer,
            use_bias=use_bias,
            use_oar=use_oar,
            oar_lm=oar_lm,
            oar_bits=oar_bits,
            s=s,
            name=name,
        ) if not cell else cell

        super(QRNNWithOAR, self).__init__(cell, return_sequences=True,
                                          stateful=stateful, unroll=to_unroll, name=name)

        # Initializers
        self.kernel_initializer = k_init
        self.recurrent_initializer = rk_init

        # Quantizers (specfically for QNoiseScheduler)
        self.quantizers = self.get_quantizers()

    def get_quantizers(self):
        return self.cell.quantizers

    def get_prunable_weights(self):
        return [self.cell.kernel, self.cell.recurrent_kernel]

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def kernel_quantizer_internal(self):
        return self.cell.kernel_quantizer_internal

    @property
    def recurrent_quantizer_internal(self):
        return self.cell.recurrent_quantizer_internal

    @property
    def bias_quantizer_internal(self):
        return self.cell.bias_quantizer_internal

    @property
    def state_quantizer_internal(self):
        return self.cell.state_quantizer_internal

    @property
    def kernel_quantizer(self):
        return self.cell.kernel_quantizer

    @property
    def recurrent_quantizer(self):
        return self.cell.recurrent_quantizer

    @property
    def bias_quantizer(self):
        return self.cell.bias_quantizer

    @property
    def state_quantizer(self):
        return self.cell.state_quantizer

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "recurrent_regularizer": self.recurrent_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "kernel_quantizer": self.kernel_quantizer,
            "recurrent_quantizer": self.recurrent_quantizer,
            "bias_quantizer": self.bias_quantizer,
        })
        return config


class QSimpleRNNCellWithOAR(QSimpleRNNCell):
    """
    Cell class for the QSimpleRNNCell layer with Overflow-Aware Activity Regularization.
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
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
                 oar_lm=0,
                 oar_bits=32,
                 s=1,
                 **kwargs):

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

        # Overflow-Aware Activity Regularization (must unroll rnn)
        self.oar = None
        self.oar_lm = oar_lm
        self.oar_bits = oar_bits
        if use_oar:
            self.oar = OAR2(
                lm=oar_lm, k=2**oar_bits, name=kwargs["name"])

        # Gradient scaling
        self.s = s

    def build(self, input_shape):
        super(QSimpleRNNCellWithOAR, self).build(input_shape)

        # For debugging to see quantized kernels
        if self.kernel_quantizer:
            self.quantized_kernel = self.add_weight(
                name="quantized_kernel",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=False
            )
            # self.kernel_qmae = tf.keras.metrics.Mean(
            #     name='qmae'+"/"+self.kernel.name)  # MAE
        if self.recurrent_quantizer:
            self.quantized_recurrent_kernel = self.add_weight(
                name="quantized_recurrent_kernel",
                shape=self.recurrent_kernel.shape,
                dtype=self.recurrent_kernel.dtype,
                initializer="zeros",
                trainable=False
            )
            # self.recurrent_kernel_qmae = tf.keras.metrics.Mean(
            #     name='qmae'+"/"+self.recurrent_kernel.name)  # MAE

        self.wx = self.add_weight(
            name="wx_abs",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False
        )

        self.wh = self.add_weight(
            name="wh_abs",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False
        )

    def call(self, inputs, states, training=None):
        prev_output = states[0] if nest.is_sequence(states) else states

        # Quantize the state
        if self.state_quantizer:
            quantized_prev_output = self.state_quantizer_internal(prev_output)
        else:
            quantized_prev_output = prev_output

        # Quantize the kernel(s)
        if self.kernel_quantizer:
            quantized_kernel = self.kernel_quantizer_internal(self.kernel)
            # For debugging to see quantized kernel
            self.quantized_kernel.assign(quantized_kernel)
            # self.add_metric(self.kernel_qmae(tf.reduce_mean(
            #     tf.abs(self.kernel-self.quantized_kernel))))
        else:
            quantized_kernel = self.kernel

        h = K.dot(inputs, quantized_kernel)

        # Quantize recurrent kernel
        if self.recurrent_quantizer:
            quantized_recurrent = self.recurrent_quantizer_internal(
                self.recurrent_kernel)
            # For debugging to see quantized kernel
            self.quantized_recurrent_kernel.assign(quantized_recurrent)
            # self.add_metric(self.recurrent_kernel_qmae(tf.reduce_mean(
            #     tf.abs(self.recurrent_kernel-self.quantized_recurrent_kernel))))
        else:
            quantized_recurrent = self.recurrent_kernel

        h_2 = K.dot(quantized_prev_output, quantized_recurrent)

        # Update stats
        self.wx.assign(0.90*self.wx + 0.10*tf.reduce_mean(tf.abs(h), axis=0))
        self.wh.assign(0.90*self.wh + 0.10*tf.reduce_mean(tf.abs(h_2), axis=0))

        # Add two dot products (W_x*x + W_h*h) with gradient scaling in backward
        # direction, not forward
        s = self.s
        output = h/s + h_2/s + tf.stop_gradient(-h/s - h_2/s + h + h_2)

        # Compute activation
        if self.activation is not None:
            # Add Overflow-Aware Activtiy Regularization
            if self.oar is not None:
                output = self.oar(output)
            output = self.activation(output)

        return output, [output]


class QDenseWithOAR(QDense):
    """Implements a quantized Dense layer WITH NORMALIZATION."""

    def __init__(self,
                 units,
                 activation=None,
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
                 oar_lm=0,
                 oar_bits=32,
                 s=1,
                 **kwargs):

        # Overflow-Aware Activity Regularization
        self.oar = None
        self.oar_lm = oar_lm
        self.oar_bits = oar_bits
        if use_oar:
            self.oar = OAR2(
                lm=oar_lm, k=2**oar_bits, name=kwargs["name"])

        # Gradient scale
        self.s = s

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
            **kwargs)

    def build(self, input_shape):
        super(QDenseWithOAR, self).build(input_shape)

        # For debugging to see quantized kernel
        if self.kernel_quantizer:
            self.quantized_kernel = self.add_weight(
                name="quantized_kernel",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=False
            )
            # self.kernel_qmae = tf.keras.metrics.Mean(
            #     name='qmae'+"/"+self.kernel.name)  # MAE

        self.wx = self.add_weight(
            name="wx_abs",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False
        )

        # wx_shp = input_shape.as_list()
        # wx_shp[0] = 512
        # wx_shp[-1] = self.units
        # self.wx_full = tf.Variable(
        #     initial_value=tf.zeros(shape=wx_shp, dtype=tf.float32),
        #     name="wx_f",
        #     trainable=False,
        #     validate_shape=False,
        # )

        self.x_input = self.add_weight(
            name="x_abs",
            shape=[input_shape[-1]],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False
        )

    def call(self, inputs):
        # Write input stats (mainly for SA mult)
        reduce_dims = tf.range(0, tf.rank(inputs)-1)
        self.x_input.assign(0.9*self.x_input + 0.1 *
                            tf.reduce_mean(tf.abs(inputs), axis=reduce_dims))

        # Quantize the kernel
        quantized_kernel = self.kernel
        if self.kernel_quantizer:
            quantized_kernel = self.kernel_quantizer_internal(self.kernel)
            # For debugging to see quantized kernel
            self.quantized_kernel.assign(quantized_kernel)
            # self.add_metric(self.kernel_qmae(tf.reduce_mean(
            #     tf.abs(self.kernel-self.quantized_kernel))))

        # Calculate Wx
        h = tf.keras.backend.dot(inputs, quantized_kernel)

        # Gradient scaling in backward direction, not forward
        s = self.s
        h = h/s + tf.stop_gradient(-h/s + h)

        # Update stats
        reduce_dims = tf.range(0, tf.rank(h)-1)
        self.wx.assign(0.9*self.wx + 0.1 *
                       tf.reduce_mean(tf.abs(h), axis=reduce_dims))
        # self.wx_full.assign(0.9*self.wx_full + 0.1*h)

        output = h
        if self.activation is not None:
            # Add Overflow-Aware Activity Regularization
            if self.oar is not None:
                output = self.oar(output)
            # Compute activation
            output = self.activation(output)
        return output


def sign_with_ste(x):
    """
    Compute the signum function in the fwd pass but return STE approximation for grad in bkwd pass
    """
    out = x
    q = tf.math.sign(x)
    q += (1.0 - tf.math.abs(q))
    return out + tf.stop_gradient(-out + q)


def sign_with_tanh_deriv(x):
    out = tf.keras.activations.tanh(x)
    q = tf.math.sign(x)
    q += (1.0 - tf.math.abs(q))
    return out + tf.stop_gradient(-out + q)


def mod_sign_with_tanh_deriv(x, num_bits=8):
    """
    Compute x mod 2**num_bits then run relu.
    Note, no gradient passes from mod op by using tf.stop_gradient
    """
    # inner function to wrap stop gradient around to not take gradient into account
    def _inner_fn(x, num_bits):
        base = 2**num_bits
        half_base = 2**(num_bits-1)

        # Cast to int to do modular reduction
        x_int = tf.cast(x, tf.int32)

        # Perform modular reduction (creating unsigned int)
        modded = tf.math.mod(x_int, base)

        # Sign the int
        signed = tf.where(tf.greater_equal(
            modded, half_base), modded-base, modded)

        # Cast back to float
        signed_float = tf.cast(signed, tf.float32)

        return signed_float

    # Regular sign
    out = sign_with_tanh_deriv(x) + tf.stop_gradient(-sign_with_tanh_deriv(
        x) + sign_with_tanh_deriv(_inner_fn(x, num_bits=num_bits)))

    # For seeing distribution of input to activation
    # (Note: The try must be kept or else it will fail on the initial tf graphing)
    # try:
    #     plot_histogram_discrete(x, "histogram_of_wxplusb.png")
    #     plot_histogram_discrete(signed_float, "histogram_of_wxplusb_modded.png")
    # except:
    #     return out
    return out


def mod_on_inputs(x, num_bits=8):
    """
    Compute x mod 2**num_bits then run relu.
    Note, no gradient passes from mod op by using tf.stop_gradient
    """
    # inner function to wrap stop gradient around to not take gradient into account
    def _inner_fn(x, num_bits):
        base = 2**num_bits
        half_base = 2**(num_bits-1)

        # Cast to int to do modular reduction
        x_int = tf.cast(x, tf.int32)

        # Perform modular reduction (creating unsigned int)
        modded = tf.math.mod(x_int, base)

        # Sign the int
        signed = tf.where(tf.greater_equal(
            modded, half_base), modded-base, modded)

        # Cast back to float
        signed_float = tf.cast(signed, tf.float32)

        return signed_float

    out = x + tf.stop_gradient(-x + _inner_fn(x, num_bits=num_bits))
    return out


class GeneralActivation(tf.keras.layers.Layer):
    def __init__(self, activation=None, name=""):
        super(GeneralActivation, self).__init__()
        self.activation = activation

        self.inp_moving_mean = self.add_weight(
            name=name+"/inp_moving_mean",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        self.inp_moving_std = self.add_weight(
            name=name+"/inp_moving_std",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        self.out_moving_mean = self.add_weight(
            name=name+"/out_moving_mean",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        self.out_moving_std = self.add_weight(
            name=name+"/out_moving_std",
            shape=[],
            dtype=tf.float32,
            initializer=tf.keras.initializers.constant(0),
            trainable=False,
        )
        self.built = False

    def build(self, input_shape):
        super(GeneralActivation, self).build(input_shape)

        # new_shape = [512, input_shape[-1]]

        # # Build shape dependent stats
        # self.input_dist = self.add_weight(
        #     name=self.name+"/pre-activations",
        #     shape=new_shape,
        #     dtype=tf.float32,
        #     initializer="zeros",
        #     trainable=False,
        # )
        # self.output_dist = self.add_weight(
        #     name=self.name+"/activations",
        #     shape=new_shape,
        #     dtype=tf.float32,
        #     initializer="zeros",
        #     trainable=False,
        # )

        self.built = True

    def __call__(self, inputs):
        input_shape = inputs.get_shape()
        size_shape = len(input_shape.as_list())
        if not self.built:
            self.build(input_shape)

        out = inputs

        # Run activation
        if self.activation:
            out = self.activation(out)

        # # Compute stats
        # inputs_for_stats = tf.zeros_like(inputs) + inputs
        # outputs_for_stats = tf.zeros_like(out) + out
        # if (size_shape == 3):
        #     inputs_for_stats = tf.reduce_mean(inputs_for_stats, 1)
        #     outputs_for_stats = tf.reduce_mean(outputs_for_stats, 1)

        # # Calculate input stats
        # self.input_dist.assign(0.1 * inputs_for_stats + 0.9 * self.input_dist)

        # # Calculate output stats
        # self.output_dist.assign(
        #     0.1 * outputs_for_stats + 0.9 * self.output_dist)

        self.inp_moving_mean.assign(
            0.9*self.inp_moving_mean + 0.1*tf.reduce_mean(inputs))
        self.inp_moving_std.assign(
            0.9*self.inp_moving_std + 0.1*tf.math.reduce_std(inputs))
        self.out_moving_mean.assign(
            0.9*self.out_moving_mean + 0.1*tf.reduce_mean(out))
        self.out_moving_std.assign(
            0.9*self.out_moving_std + 0.1*tf.math.reduce_std(out))

        return out

    def get_config(self):
        return {
            "alpha_init": self.alpha_init,
        }


def oar_hat_fn(x, k, a):
    t_x = 1/k*(tf.abs(x)-(3/4*k-1/2))
    mod = tf.math.mod(t_x, 1)
    abs = tf.abs(2*mod-1)
    out = 2*abs-1
    return a*tf.nn.relu(out)


def oar_hat_metric_fn(x, k, a):
    """
    A function that calculates the ratio between values in correct regions and all values.
    To be used as input into metric function (ex. tf.keras.metrics.Mean()).

    Takes the sign of the no_acc_reg function which gives you whether a value is in a wrong
    region (+1) or correct region (0). Adding all the values in the tensor and dividing by the
    total number of values is effectively the count of wrong values over all values. This is 
    also more easily represented as the mean of the tensor. 1 - mean(wrongs) gives the ratio
    of correct values over all values.

    k is the modulus of quantization.
    a is the height of no_acc_reg_hat_fn.

    Important: this implementation only to be used with no_acc_reg_hat_fn!
    """
    wrongs = tf.sign(oar_hat_fn(x, k=k, a=a))
    rights_ratio = 1 - tf.reduce_mean(wrongs, axis=[-1])
    return rights_ratio


class OAR1(tf.keras.layers.Layer):
    """
    This regularizer penalizes pre-activations that lie in incorrectly-signed regions of the domain
    after a modulo operation and sign are applied.

    Ex. if K (modulus) = 8, then a result of Wx_i mod 8 = 4 mod 8 = -4 instead of +4 which can lead a 
    sign activation function outputting wrong -1 instead of +1. This is important for binary/ternary
    networks.

    This specific version implements a hat function (ex. __/\__/\__). This is supposed to provide a constant
    derivative in the incorrect regions to move them to the correct regions.
    """

    def __init__(self, lm=1e-3, k=2**8, a=1., name=""):
        super(OAR1, self).__init__()
        self.lm = lm
        self.k = k
        self.a = a
        self.no_acc_metric = tf.keras.metrics.Mean(name='OAR1/'+name)

    def __call__(self, x):
        loss = oar_hat_fn(x=x, k=self.k, a=self.a)
        loss = self.lm * tf.reduce_sum(loss)

        accuracy = oar_hat_metric_fn(x, k=self.k, a=self.a)
        accuracy = self.no_acc_metric(accuracy)

        self.add_loss(loss)
        self.add_metric(accuracy)

        return x

    def get_config(self):
        return {'lm': float(self.lm), 'k': int(self.k), 'a': float(self.a)}


class OAR2(tf.keras.layers.Layer):
    """
    This regularizer penalizes pre-activations that lie in incorrectly-signed regions of the domain
    after a modulo operation and sign are applied. The output is squared.

    Ex. if K (modulus) = 8, then a result of Wx_i mod 8 = 4 mod 8 = -4 instead of +4 which can lead a 
    sign activation function outputting wrong -1 instead of +1. This is important for binary/ternary
    networks.

    This specific version implements a hat function (ex. __/\__/\__). This is supposed to provide a constant
    derivative in the incorrect regions to move them to the correct regions.
    """

    def __init__(self, lm=1e-3, k=2**8, a=1., name=""):
        super(OAR2, self).__init__()
        self.lm = lm
        self.k = k
        self.a = a
        self.no_acc_metric = tf.keras.metrics.Mean(name='OAR2/'+name)

    def __call__(self, x):
        loss = tf.square(oar_hat_fn(x=x, k=self.k, a=self.a))
        loss = self.lm * tf.reduce_sum(loss)

        accuracy = oar_hat_metric_fn(x, k=self.k, a=self.a)
        accuracy = self.no_acc_metric(accuracy)

        self.add_loss(loss)
        self.add_metric(accuracy)

        return x

    def get_config(self):
        return {'lm': float(self.lm), 'k': int(self.k), 'a': float(self.a)}


class BreakpointLayerForDebug(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BreakpointLayerForDebug, self).__init__(**kwargs)

    def call(self, inputs):
        vele = "ferus"  # place breakpoint here
        # inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        return inputs


class TimeReduction(tf.keras.layers.Layer):

    def __init__(self,
                 reduction_factor,
                 batch_size=None,
                 **kwargs):

        super(TimeReduction, self).__init__(**kwargs)

        self.reduction_factor = reduction_factor
        self.batch_size = batch_size

    def compute_output_shape(self, input_shape):
        max_time = input_shape[1]
        num_units = input_shape[2]
        if max_time != None:  # For time variance
            extra_timestep = tf.math.floormod(max_time, self.reduction_factor)
            reduced_size = tf.math.floordiv(
                max_time, self.reduction_factor) + extra_timestep
        else:
            reduced_size = None
        return [input_shape[0], reduced_size, num_units*self.reduction_factor]

    def call(self, inputs):

        input_shape = K.int_shape(inputs)

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = input_shape[0]

        outputs = inputs

        if input_shape[1] != None:
            max_time = input_shape[1]
            extra_timestep = tf.math.floormod(max_time, self.reduction_factor)

            paddings = [[0, 0], [0, extra_timestep], [0, 0]]
            outputs = tf.pad(outputs, paddings)

        else:
            outputs = tf.signal.frame(
                outputs, self.reduction_factor, self.reduction_factor, pad_end=False, axis=1)

        # Necessary to let tf know correct output shape
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


class ModelWithGradInfo(tf.keras.models.Model):
    def train_step(self, data):
        """The logic for one training step.

        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happens in fit](
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.

        This method should contain the mathematical logic for one step of
        training.  This typically includes the forward pass, loss calculation,
        backpropagation, and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function`
        and `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Args:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        output = self.compute_metrics(x, y, y_pred, sample_weight)

        # Calc additional grad stats
        # REG GRADS
        # reg_grads = tape.gradient(tf.add_n(self.losses), self.trainable_variables)
        # reg_grad_norms_names = ["grad_norm/reg_"+g.name for g in self.trainable_variables]
        # reg_grad_squares = [tf.reduce_sum(tf.square(g)) for g in reg_grads]
        # reg_grad_norms = [tf.sqrt(g) for g in reg_grad_squares]
        # reg_global_grad_norm = tf.sqrt(tf.add_n(reg_grad_squares))
        # reg_names_to_norms = dict(zip(reg_grad_norms_names, reg_grad_norms))

        # output["global_reg_grad_norm"] =  reg_global_grad_norm
        # output.update(reg_names_to_norms)

        # TOTAL GRADS
        # grad_norms_names = ["grad_norm/" +
        #                     g.name for g in self.trainable_variables]
        # grad_avgs_names = ["grad_avg/"+g.name for g in self.trainable_variables]
        # grad_squares = [tf.reduce_sum(tf.square(g)) for g in grads]
        # grad_avgs = [tf.reduce_mean(g) for g in grads]
        # grad_norms = [tf.sqrt(g) for g in grad_squares]

        # global_grad_norm = tf.sqrt(tf.add_n(grad_squares))
        # names_to_norms = dict(zip(grad_norms_names, grad_norms))
        # names_to_avgs = dict(zip(grad_avgs_names, grad_avgs))

        # Update output dict
        # output["global_grad_norm"] = global_grad_norm
        # output.update(names_to_norms)
        # output.update(names_to_avgs)

        return output


def custom_loader(model, checkpoint_path):
    # Load the weights from the checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path).expect_partial()

    # Get a list of all model's variables
    model_vars = model.variables

    # Iterate over the variables
    for var in model_vars:
        # Get the name of the current variable
        var_name = var.name

        # Skip the loading of certain variables based on their name
        if "wx" in var_name:
            continue

        # Load the value of the variable
        value = checkpoint.get_variable_value(var_name)
        var.assign(value)
