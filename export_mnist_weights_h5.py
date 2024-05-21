import os
from utils.model_utils import get_default_layer_options_from_options
from mnist_rnn_model import get_model


def export_mnist_weights(pretrained_weights: str, options):
    """Given the pretrained weights folder, extract the weights in hdf5 format and save
    in the same folder, in subfolder `hdf5`.

    Args:
        pretrained_weights (str): path to checkpoints folder containing model parameters
        options: options for model
    """
    model = get_model(options, get_default_layer_options_from_options(options))

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
