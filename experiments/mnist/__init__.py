"""MNIST RNN experiment with OAR regularization."""

from experiments.mnist.data import get_datasets, normalize_img, resize, augment
from experiments.mnist.model import get_model
from experiments.mnist.config import (
    perform_training_steps,
    get_model_parameter_stats_new,
)
from experiments.mnist.steps import STATIC_STEPS, create_step_4
from experiments.mnist.training import (
    configure_environment,
    train,
    create_run_dir,
    tee_output,
)

__all__ = [
    "get_datasets",
    "normalize_img",
    "resize",
    "augment",
    "get_model",
    "perform_training_steps",
    "get_model_parameter_stats_new",
    "STATIC_STEPS",
    "create_step_4",
    "configure_environment",
    "train",
    "create_run_dir",
    "tee_output",
]
