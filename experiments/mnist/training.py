"""MNIST training utilities."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING

import tensorflow as tf

from oar import ReservoirHistogramCallback, reset_stat_weights

if TYPE_CHECKING:
    from oar.config import TrainingStepConfig

RUNS_DIR = "runs/mnist/"
TB_LOGS_DIR = "logs/tensorboard/"
CKPT_DIR = "checkpoints/"
RECORD_CKPTS = True


def create_run_dir() -> str:
    """Creates a new timestamped run directory.

    Returns:
        Path to the created run directory (with trailing slash)
    """
    now = datetime.now()
    run_dir = RUNS_DIR + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


@contextmanager
def tee_output(log_path: str):
    """Context manager to tee stdout/stderr to a log file.

    Writes all output to both the console and a log file.

    Args:
        log_path: Path to log file

    Yields:
        None
    """

    class TeeWriter:
        """File-like object that writes to both original stream and log file."""

        def __init__(self, original, log_file):
            self.original = original
            self.log_file = log_file
            self.encoding = getattr(original, "encoding", "utf-8")

        def write(self, message):
            self.original.write(message)
            self.log_file.write(message)
            self.log_file.flush()

        def flush(self):
            self.original.flush()
            self.log_file.flush()

        def isatty(self):
            return False  # Log file is not a tty

        def fileno(self):
            return self.original.fileno()

    with open(log_path, "w") as log_file:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = TeeWriter(old_stdout, log_file)
        sys.stderr = TeeWriter(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _save_step_config(
    step_dir: str,
    step_number: int | None,
    step_config: "TrainingStepConfig",
    pretrained_weights: str | None,
) -> None:
    """Save configuration for a training step.

    Args:
        step_dir: Directory to save config to
        step_number: Step number (1-4) or None
        step_config: Training step configuration
        pretrained_weights: Path to pretrained weights
    """
    from dataclasses import asdict

    config = {
        "step": step_number,
        "step_config": asdict(step_config),
        "pretrained_weights": pretrained_weights,
    }

    config_path = os.path.join(step_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def configure_environment():
    """Configures the environment by selecting the datatype and device strategy.

    Returns:
        Tuple of (strategy, dtype)
    """
    dtype = tf.float32

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(str(e))

        print(gpus)
        device = gpus[0].name[17:]
        print("Running single gpu: {}".format(device))
        strategy = tf.distribute.OneDeviceStrategy(device=device)
    else:
        device = tf.config.list_physical_devices("CPU")[0].name[17:]
        print("Running on CPU: {}".format(device))
        strategy = tf.distribute.OneDeviceStrategy(device=device)

    return strategy, dtype


def train(
    step_config: "TrainingStepConfig",
    step_number: int | None = None,
    pretrained_weights: str | None = None,
    run_dir: str | None = None,
) -> str:
    """Run training for a step.

    Args:
        step_config: Training step configuration (contains all params)
        step_number: Step number for logging (1-indexed)
        pretrained_weights: Path to checkpoints folder
        run_dir: Run directory (creates new if None)

    Returns:
        Path to checkpoints folder
    """
    from experiments.mnist.model import get_model
    from experiments.mnist.data import get_datasets

    strategy, _ = configure_environment()

    if run_dir is None:
        run_dir = create_run_dir()

    if step_number is not None:
        output_dir = os.path.join(run_dir, f"step_{step_number}") + "/"
    else:
        output_dir = run_dir

    os.makedirs(output_dir, exist_ok=True)
    _save_step_config(output_dir, step_number, step_config, pretrained_weights)

    ds_train, ds_val, ds_test = get_datasets(
        batch_size=step_config.batch_size,
        enlarge=step_config.enlarge,
    )

    with strategy.scope():
        model = get_model(step_config)

        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)
            print(f"Restored pretrained weights from {pretrained_weights}.")

        reset_stat_weights(model)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=step_config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    # Callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=output_dir + TB_LOGS_DIR,
        histogram_freq=1,
        update_freq="epoch",
    )
    reservoir_cb = ReservoirHistogramCallback(log_dir=output_dir + TB_LOGS_DIR)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        tf.keras.optimizers.schedules.CosineDecay(
            step_config.learning_rate, 100, alpha=0.1
        ),
        verbose=0,
    )

    ckpt_callback = None
    if RECORD_CKPTS:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir + CKPT_DIR,
            save_weights_only=True,
            save_best_only=False,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        )

    model.fit(
        ds_train,
        epochs=step_config.epochs,
        validation_data=ds_val,
        callbacks=[cb for cb in [tb_callback, reservoir_cb, ckpt_callback, lr_callback] if cb],
        verbose=2,
    )

    print("\nRUNNING EVALUATION OVER TEST SET\n")
    model.evaluate(ds_test, verbose=2)

    return output_dir + CKPT_DIR
