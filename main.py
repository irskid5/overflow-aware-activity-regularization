"""Entry point for OAR experiments."""

import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit("autoclustering")

tf.random.set_seed(1997)  # For experimental reproducibility

from oar.runner import Runner
from experiments.mnist.config import MNIST_EXPERIMENT
from experiments.mnist.model import get_model
from experiments.mnist.data import get_datasets


def main():
    """Runs four-step quantization using the new Runner API."""
    runner = Runner(
        experiment=MNIST_EXPERIMENT,
        model_factory=get_model,
        data_loader=get_datasets,
    )
    checkpoint = runner.run()
    print(f"\nTraining complete. Final checkpoint: {checkpoint}")
    return checkpoint


def run_from_step_3(checkpoint_path: str) -> str:
    """Resume training from step 3.

    Args:
        checkpoint_path: Path to step 2 checkpoint (.keras format).

    Returns:
        Path to final checkpoint from the completed run.
    """
    runner = Runner(
        experiment=MNIST_EXPERIMENT,
        model_factory=get_model,
        data_loader=get_datasets,
    )
    return runner.run(start_step=3, resume_from=checkpoint_path)


def run_from_step_4(checkpoint_path: str) -> str:
    """Resume training from step 4 (full quantization).

    Args:
        checkpoint_path: Path to step 3 checkpoint (e.g. runs/mnist/<run_id>/step_3/checkpoints/).

    Returns:
        Path to final checkpoint from the completed run.
    """
    runner = Runner(
        experiment=MNIST_EXPERIMENT,
        model_factory=get_model,
        data_loader=get_datasets,
    )
    return runner.run(start_step=4, end_step=4, resume_from=checkpoint_path)


if __name__ == "__main__":
    main()
    print("End!")
