import os

import tensorflow as tf
from tensorflow import keras

from mnist_rnn_model import get_model

# DEFAULT PARAMETERS FROM STEP 1 (DON'T TOUCH, ALSO UNNECESSARY FOR EXTRACTING WEIGHTS)
ternarize_inputs = False
t = 1.0
s = 1.0
activation = tf.keras.activations.tanh
oar = False
tern_params = {"QRNN_0": 0, "QRNN_1": 0, "DENSE_0": 0, "DENSE_OUT": 0}
options = {
    "enlarge": False,
    "epochs": 10,
    "learning_rate": 5e-5,
    "batch_size": 512,
    "t": 1.5,
    "tᵢ": 0.7,
    "s": 4.0,
    "oar": {
        "lm": 1e-4,
        "precision": 6,
    },
    "quantize": False,
}
layer_options = {
    "INPUT": {"ternarize": ternarize_inputs},
    "QRNN_0": {
        "activation": activation,
        "oar": {
            "use": oar,
            "lm": options["oar"]["lm"],
            "precision": options["oar"]["precision"],
        },
        "s": s,
        "τ": t * tern_params["QRNN_0"],
    },
    "QRNN_1": {
        "activation": activation,
        "oar": {
            "use": oar,
            "lm": options["oar"]["lm"],
            "precision": options["oar"]["precision"],
        },
        "s": s,
        "τ": t * tern_params["QRNN_1"],
    },
    "DENSE_0": {
        "activation": activation,
        "oar": {
            "use": oar,
            "lm": options["oar"]["lm"],
            "precision": options["oar"]["precision"],
        },
        "s": 1.0,
        "τ": t * tern_params["DENSE_0"],
    },
    "DENSE_OUT": {
        "activation": lambda x: tf.keras.activations.softmax(x),
        "oar": {
            "use": True,
            "lm": 0.0,
            "precision": options["oar"]["precision"],
        },
        "s": 1.0,
        "τ": t * tern_params["DENSE_OUT"],
    },
}


def export_mnist_weights(pretrained_weights: str):
    """Given the pretrained weights folder, extract the weights in hdf5 format and save
    in the same folder, in subfolder `hdf5`.

    Args:
        pretrained_weights (str): path to checkpoints folder containing model parameters
    """
    model = get_model(options, layer_options)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
        print("Restored pretrained weights from {}.".format(pretrained_weights))

    # Reset the stat variables
    weights = model.get_weights()
    weights_to_discard = []
    for i in range(len(weights)):
        if (
            "/w" in model.weights[i].name
            or "/x" in model.weights[i].name
            or "preacts" in model.weights[i].name
        ):
            weights[i] = 0 * weights[i]
    model.set_weights(weights)

    h5_dir = pretrained_weights + "hdf5/"
    if not os.path.exists(h5_dir):
        os.makedirs(h5_dir)
    h5_filepath = h5_dir + "weights.hdf5"

    print("Saving weights from -> " + pretrained_weights)
    print("Saving weights to   -> " + h5_filepath)
    model.save_weights(h5_filepath, overwrite=False, save_format="h5")
    print("Completed. Goodbye.")
