from qkeras import *

from utils.model_utils import TimeReduction, QDenseWithOAR, QRNNWithOAR, ModelWithGradInfo, GeneralActivation
from quantization import ternarize_tensor_with_threshold, LearnedThresholdTernary

SEED = 1997

# Regularizers
kernel_regularizer = None
recurrent_regularizer = None
bias_regularizer = None
activation_regularizer = None

# Initializers
rnn_kernel_initializer = tf.keras.initializers.VarianceScaling(
    scale=1.0, mode="fan_avg", distribution="uniform", seed=SEED)
rnn_recurrent_initializer = tf.keras.initializers.Orthogonal(
    gain=1.0, seed=SEED)
dense_kernel_initializer = tf.keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_in", distribution="truncated_normal", seed=SEED)


def get_model(options, layer_options):
    input = tf.keras.layers.Input(
        shape=(28, 28, 1) if not options["enlarge"] else (128, 128, 1))

    # Ternarize inputs (if step 3)
    input = tf.keras.layers.Lambda(
        lambda x: tf.stop_gradient(ternarize_tensor_with_threshold(
            x, theta=options["tᵢ"]*tf.reduce_mean(tf.abs(x)))),
        trainable=False,
        dtype=tf.float32,
        name="TERNARIZE_WITH_THRESHOLD"
    )(input) if layer_options["INPUT"]["ternarize"] else tf.keras.layers.Lambda(lambda x: x, name="NOOP")(input)

    input = tf.keras.layers.Reshape(target_shape=(
        28, 28) if not options["enlarge"] else (128, 128))(input)

    qrnn_0 = QRNNWithOAR(
        cell=None,
        units=128,
        activation=GeneralActivation(
            activation=layer_options["QRNN_0"]["activation"], name="QRNN_0"),
        use_bias=False,
        return_sequences=True,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0,
            threshold=layer_options["QRNN_0"]["τ"],
            name="QRNN_0/quantized_kernel") if options["quantize"] else None,
        recurrent_quantizer=LearnedThresholdTernary(
            scale=1.0,
            threshold=layer_options["QRNN_0"]["τ"],
            name="QRNN_0/quantized_recurrent") if options["quantize"] else None,
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        use_oar=layer_options["QRNN_0"]["oar"]["use"],
        oar_lm=layer_options["QRNN_0"]["oar"]["lm"],
        oar_bits=layer_options["QRNN_0"]["oar"]["precision"],
        s=layer_options["QRNN_0"]["s"],
        name="QRNN_0")(input)
    tr = TimeReduction(reduction_factor=2)(qrnn_0)
    qrnn_1 = QRNNWithOAR(
        cell=None,
        units=128,
        activation=GeneralActivation(
            activation=layer_options["QRNN_1"]["activation"], name="QRNN_1"),
        use_bias=False,
        return_sequences=True,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0,
            threshold=layer_options["QRNN_1"]["τ"],
            name="QRNN_1/quantized_kernel") if options["quantize"] else None,
        recurrent_quantizer=LearnedThresholdTernary(
            scale=1.0,
            threshold=layer_options["QRNN_1"]["τ"],
            name="QRNN_1/quantized_recurrent") if options["quantize"] else None,
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        use_oar=layer_options["QRNN_1"]["oar"]["use"],
        oar_lm=layer_options["QRNN_1"]["oar"]["lm"],
        oar_bits=layer_options["QRNN_1"]["oar"]["precision"],
        s=layer_options["QRNN_1"]["s"],
        name="QRNN_1")(tr)
    qrnn_1 = tf.keras.layers.Flatten()(qrnn_1)
    dense_0 = QDenseWithOAR(
        1024,
        activation=GeneralActivation(
            activation=layer_options["DENSE_0"]["activation"], name="DENSE_0"),
        use_bias=False,
        kernel_regularizer=kernel_regularizer,
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0,
            threshold=layer_options["DENSE_0"]["τ"],
            name="DENSE_0") if options["quantize"] else None,
        kernel_initializer=dense_kernel_initializer,
        use_oar=layer_options["DENSE_0"]["oar"]["use"],
        oar_lm=layer_options["DENSE_0"]["oar"]["lm"],
        oar_bits=layer_options["DENSE_0"]["oar"]["precision"],
        s=layer_options["DENSE_0"]["s"],
        name="DENSE_0",)(qrnn_1)
    output = QDenseWithOAR(
        10,
        use_bias=False,
        activation=GeneralActivation(
            activation=layer_options["DENSE_OUT"]["activation"], name="DENSE_OUT"),
        kernel_regularizer=kernel_regularizer,
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0,
            threshold=layer_options["DENSE_OUT"]["τ"],
            name="DENSE_OUT") if options["quantize"] else None,
        kernel_initializer=dense_kernel_initializer,
        use_oar=layer_options["DENSE_OUT"]["oar"]["use"],
        oar_lm=layer_options["DENSE_OUT"]["oar"]["lm"],
        oar_bits=layer_options["DENSE_OUT"]["oar"]["precision"],
        s=layer_options["DENSE_OUT"]["s"],
        name="DENSE_OUT",)(dense_0)

    model = ModelWithGradInfo(inputs=[input],
                              outputs=[output], name="MNIST_RNN" if not options["enlarge"] else "ENLARGED_MNIST_RNN")

    model.summary()

    return model
