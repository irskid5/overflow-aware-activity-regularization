import os
from oar import reset_stat_weights
from oar.config import TrainingStepConfig
from experiments.mnist import get_model


def export_mnist_weights(pretrained_weights: str, enlarge: bool = False):
    """Given the pretrained weights folder, extract the weights in hdf5 format.

    Args:
        pretrained_weights: Path to checkpoints folder containing model parameters
        enlarge: Whether model uses enlarged (128x128) input
    """
    # Create a minimal step config just for model construction
    step_config = TrainingStepConfig(name="export", enlarge=enlarge)
    model = get_model(step_config)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
        print("Restored pretrained weights from {}.".format(pretrained_weights))

    # Reset the stat variables
    reset_stat_weights(model)

    h5_dir = pretrained_weights + "hdf5/"
    if not os.path.exists(h5_dir):
        os.makedirs(h5_dir)
    h5_filepath = h5_dir + "weights.hdf5"

    print("Saving weights from -> " + pretrained_weights)
    print("Saving weights to   -> " + h5_filepath)
    model.save_weights(h5_filepath, overwrite=False, save_format="h5")
    print("Completed. Goodbye.")
