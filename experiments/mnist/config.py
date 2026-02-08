"""MNIST experiment configuration and training orchestration."""

import tensorflow as tf
from oar import mod_sign, sign_ste_tanh

# Default options for MNIST RNN experiment
MNIST_OPTIONS = {
    "enlarge": False,
    "epochs": 100,
    "learning_rate": 1e-4,
    "batch_size": 512,
    "t": 1.5,  # Ternarization threshold multiplier
    "tᵢ": 0.7,  # Input ternarization threshold
    "s": 4.0,  # Gradient scaling factor
    "oar": {
        "oar_lambda": 1e-4,  # OAR regularization rate
        "omega": 6,  # Bit precision (2^omega modulus)
    },
    "quantize": False,
}

# Enlarged MNIST options (128x128)
ENLARGED_MNIST_OPTIONS = {
    "enlarge": True,
    "epochs": 100,
    "learning_rate": 5e-6,
    "batch_size": 512,
    "t": 1.5,
    "tᵢ": 0.7,
    "s": 4.0,
    "oar": {
        "oar_lambda": 1e-4,
        "omega": 6,
    },
    "quantize": False,
}


def get_default_layer_options(options):
    """Build default layer_options dict from high-level options.

    Creates per-layer configuration for MNIST RNN architecture.
    Used as starting point for four-step quantization.

    Args:
        options: Dict with "oar" sub-dict containing "omega" and "oar_lambda"

    Returns:
        Dict mapping layer names to their configuration
    """
    def make_oar_config(use: bool = False, oar_lambda: float | None = None):
        return {
            "use": use,
            "oar_lambda": oar_lambda if oar_lambda is not None else options["oar"]["oar_lambda"],
            "omega": options["oar"]["omega"],
        }

    result = {"INPUT": {"ternarize": False}}

    for name in ["QRNN_0", "QRNN_1", "DENSE_0"]:
        result[name] = {
            "activation": tf.keras.activations.tanh,
            "oar": make_oar_config(),
            "s": 1.0,
            "τ": 0.0,
        }

    result["DENSE_OUT"] = {
        "activation": tf.keras.activations.softmax,
        "oar": make_oar_config(use=True, oar_lambda=0.0),
        "s": 1.0,
        "τ": 0.0,
    }

    return result


def get_model_parameter_stats(pretrained_weights: str | None, options, layer_options):
    """Gets the mean of the absolute value for ternarization of each parameter.

    Used to compute per-layer ternarization thresholds: τ = t * E(|θ_l|)

    Args:
        pretrained_weights: Path to checkpoints folder containing parameters
        options: Model options
        layer_options: Layer-wise model options

    Returns:
        Dict mapping layer names to mean absolute weight values
    """
    from experiments.mnist.model import get_model

    tern_params = {"QRNN_0": 0, "QRNN_1": 0, "DENSE_0": 0, "DENSE_OUT": 0}
    model = get_model(options, layer_options)

    if pretrained_weights is None:
        return tern_params

    model.load_weights(pretrained_weights)
    print("Restored pretrained weights from {}.".format(pretrained_weights))

    # Calculate weight stats per layer
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:
            all_weights = tf.concat(
                [tf.reshape(x, shape=[-1]) for x in layer.trainable_weights], axis=-1
            )
            mean_abs = tf.math.reduce_mean(tf.abs(all_weights))
            if layer.name.find("QRNN_0") != -1:
                tern_params["QRNN_0"] = mean_abs.numpy()
            if layer.name.find("QRNN_1") != -1:
                tern_params["QRNN_1"] = mean_abs.numpy()
            if layer.name.find("DENSE_0") != -1:
                tern_params["DENSE_0"] = mean_abs.numpy()
            if layer.name.find("DENSE_OUT") != -1:
                tern_params["DENSE_OUT"] = mean_abs.numpy()
    return tern_params


def perform_step_in_four_step_quant(step: int, pretrained_weights: str | None, options) -> str:
    """Performs a step in the four-step quantization procedure.

    Steps:
    1. Train with tanh activation (baseline)
    2. Replace tanh with sign (tanh gradient), apply gradient scaling s
    3. Ternarize inputs
    4. Enable weight quantization + OAR (learning rate ×0.1)

    Args:
        step: Step number (1-4)
        pretrained_weights: Path to checkpoints folder for initialization
        options: Overall options dict

    Returns:
        Path to checkpoints folder of the trained model
    """
    # Import here to avoid circular imports
    from experiments.mnist.training import train

    print(f"\nPERFORMING STEP {step}/4 FROM FOUR-STEP QUANTIZATION PROCESS\n")

    layer_options = get_default_layer_options(options)

    # Change settings according to step number
    ternarize_inputs = False
    t = 1.0
    s = 1.0
    activation = tf.keras.activations.tanh
    oar = False
    tern_params = {"QRNN_0": 0, "QRNN_1": 0, "DENSE_0": 0, "DENSE_OUT": 0}

    if step == 2:
        s = options["s"]
        activation = sign_ste_tanh
    if step == 3:
        ternarize_inputs = True
    if step == 4:
        options["quantize"] = True
        options["learning_rate"] *= 0.1
        t = options["t"]
        oar = True
        tern_params = get_model_parameter_stats(
            pretrained_weights, options, layer_options
        )
        print("\nTERNARIZATION PARAMETERS:")
        print(tern_params)

        def activation(x):
            return mod_sign(x, num_bits=options["oar"]["omega"])

    # Adjust layer options
    def make_oar_config(use: bool, oar_lambda: float):
        return {
            "use": use,
            "oar_lambda": oar_lambda,
            "omega": options["oar"]["omega"],
        }

    layer_options = {"INPUT": {"ternarize": ternarize_inputs}}

    for name in ["QRNN_0", "QRNN_1"]:
        layer_options[name] = {
            "activation": activation,
            "oar": make_oar_config(oar, options["oar"]["oar_lambda"]),
            "s": s,
            "τ": t * tern_params[name],
        }

    layer_options["DENSE_0"] = {
        "activation": activation,
        "oar": make_oar_config(oar, options["oar"]["oar_lambda"]),
        "s": 1.0,
        "τ": t * tern_params["DENSE_0"],
    }

    layer_options["DENSE_OUT"] = {
        "activation": tf.keras.activations.softmax,
        "oar": make_oar_config(use=True, oar_lambda=0.0),
        "s": 1.0,
        "τ": t * tern_params["DENSE_OUT"],
    }

    print("OPTIONS AND LAYER OPTIONS:")
    print(options)
    print(layer_options)

    return train(
        pretrained_weights=pretrained_weights,
        options=options,
        layer_options=layer_options,
    )


def perform_four_step_quant(options) -> str:
    """Performs the complete four-step quantization procedure.

    Args:
        options: Experiment options dict

    Returns:
        Path to checkpoint folder of final step parameters
    """
    pretrained_weights = None
    for step in range(1, 5):
        pretrained_weights = perform_step_in_four_step_quant(
            step=step, pretrained_weights=pretrained_weights, options=options
        )
        options["epochs"] = 1000
    return pretrained_weights
