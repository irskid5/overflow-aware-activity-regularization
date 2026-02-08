"""Entry point for OAR experiments."""

import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit("autoclustering")

tf.random.set_seed(1997)  # For experimental reproducibility

from experiments.mnist import (
    MNIST_OPTIONS,
    ENLARGED_MNIST_OPTIONS,
    perform_four_step_quant,
    perform_step_in_four_step_quant,
    get_default_layer_options,
    get_model_parameter_stats,
)
from export_mnist_weights_h5 import export_mnist_weights
from export_mnist import extract_ternarized_mnist_test_dataset


def train_quantize_extract_MNIST_RNN() -> str:
    """Runs four-step quantization and exports weights for standard MNIST."""
    import copy
    options = copy.deepcopy(MNIST_OPTIONS)
    final_parameters = perform_four_step_quant(options)
    export_mnist_weights(final_parameters, options)
    return final_parameters


def train_quantize_extract_enlarged_MNIST_RNN() -> str:
    """Runs four-step quantization and exports weights for enlarged MNIST."""
    import copy
    options = copy.deepcopy(ENLARGED_MNIST_OPTIONS)
    final_parameters = perform_four_step_quant(options)
    export_mnist_weights(final_parameters, options)
    return final_parameters


def evaluation_with_and_without_oar2():
    """Evaluates models with and without OAR2 across different bit widths."""
    import copy
    third_step_options = copy.deepcopy(MNIST_OPTIONS)
    third_step = None
    for step in range(1, 4):
        third_step = perform_step_in_four_step_quant(
            step=step, pretrained_weights=third_step, options=third_step_options
        )
        third_step_options["epochs"] = 1000
    for i in range(2):
        for ω in range(3, 9):
            cur_options = copy.deepcopy(third_step_options)
            cur_options["oar"]["omega"] = ω
            cur_options["oar"]["oar_lambda"] = i * 1e-3
            perform_step_in_four_step_quant(
                step=4, pretrained_weights=third_step, options=cur_options
            )


def evaluation_different_oar_regularization_rates():
    """Evaluates different OAR regularization rates."""
    import copy
    third_step_options = copy.deepcopy(MNIST_OPTIONS)
    third_step = None
    for step in range(1, 4):
        third_step = perform_step_in_four_step_quant(
            step=step, pretrained_weights=third_step, options=third_step_options
        )
        third_step_options["epochs"] = 1000
    rates = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    bits = [5, 6]
    for ω in bits:
        for oar_lambda in rates:
            cur_options = copy.deepcopy(third_step_options)
            cur_options["oar"]["omega"] = ω
            cur_options["oar"]["oar_lambda"] = oar_lambda
            perform_step_in_four_step_quant(
                step=4, pretrained_weights=third_step, options=cur_options
            )


if __name__ == "__main__":
    train_quantize_extract_MNIST_RNN()
    # train_quantize_extract_enlarged_MNIST_RNN()
    # evaluation_with_and_without_oar2()
    # evaluation_different_oar_regularization_rates()
    # extract_ternarized_mnist_test_dataset()
    print("End!")
