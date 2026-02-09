"""Entry point for OAR experiments."""

import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit("autoclustering")

tf.random.set_seed(1997)  # For experimental reproducibility

from experiments.mnist import perform_training_steps, STATIC_STEPS, create_step_4


def main():
    """Runs four-step quantization using the new config-driven API."""
    final_checkpoint = perform_training_steps(
        steps=STATIC_STEPS,
        step_4_factory=create_step_4,
    )
    print(f"\nTraining complete. Final checkpoint: {final_checkpoint}")
    return final_checkpoint


if __name__ == "__main__":
    main()
