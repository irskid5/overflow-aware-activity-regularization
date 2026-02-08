"""MNIST training utilities."""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime

import tensorflow as tf

from oar import ReservoirHistogramCallback, reset_stat_weights
from experiments.mnist.data import get_datasets
from experiments.mnist.model import get_model

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


def _serialize_layer_options(layer_options: dict) -> dict:
    """Convert layer_options to JSON-serializable format.

    Replaces activation functions with their string names.

    Args:
        layer_options: Layer options dict with activation functions

    Returns:
        JSON-serializable dict
    """
    result = {}
    for layer_name, layer_config in layer_options.items():
        if isinstance(layer_config, dict):
            serialized = dict(layer_config)  # shallow copy
            if "activation" in serialized and callable(serialized["activation"]):
                func = serialized["activation"]
                serialized["activation"] = getattr(func, "__name__", repr(func))
            result[layer_name] = serialized
        else:
            result[layer_name] = layer_config
    return result


def _save_step_config(
    step_dir: str,
    step: int | None,
    options: dict,
    layer_options: dict,
    pretrained_weights: str | None,
) -> None:
    """Save configuration for a training step.

    Args:
        step_dir: Directory to save config to
        step: Step number (1-4) or None
        options: Training options
        layer_options: Per-layer options
        pretrained_weights: Path to pretrained weights
    """
    config = {
        "step": step,
        "options": options,
        "layer_options": _serialize_layer_options(layer_options),
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
    pretrained_weights: str | None,
    options: dict,
    layer_options: dict,
    step: int | None = None,
    run_dir: str | None = None,
) -> str:
    """Runs training for a number of epochs.

    Loads pretrained weights, builds model, and runs training.
    Also runs evaluation over test set at the end.

    Args:
        pretrained_weights: Path to checkpoints folder for initialization
        options: Training options dict
        layer_options: Per-layer options dict
        step: Optional step number for four-step quantization. When provided,
            creates a step_N subdirectory within run_dir.
        run_dir: Optional path to run directory. If None, creates a new
            timestamped directory using create_run_dir().

    Returns:
        Path to checkpoints folder of trained model
    """
    strategy, _ = configure_environment()

    # Determine run directory
    if run_dir is None:
        run_dir = create_run_dir()

    # Determine output directory (with optional step subdirectory)
    if step is not None:
        output_dir = os.path.join(run_dir, f"step_{step}") + "/"
    else:
        output_dir = run_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    _save_step_config(output_dir, step, options, layer_options, pretrained_weights)

    BATCHSIZE = options["batch_size"]
    ds_train, ds_val, ds_test = get_datasets(
        batch_size=BATCHSIZE, enlarge=options["enlarge"]
    )

    with strategy.scope():
        model = get_model(options, layer_options)

        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)
            print("Restored pretrained weights from {}.".format(pretrained_weights))

        reset_stat_weights(model)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=options["learning_rate"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    # TensorBoard callback
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=(output_dir + TB_LOGS_DIR),
        histogram_freq=1,
        update_freq="epoch",
    )
    reservoir_cb = ReservoirHistogramCallback(log_dir=(output_dir + TB_LOGS_DIR))

    # Learning rate schedule
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        tf.keras.optimizers.schedules.CosineDecay(
            options["learning_rate"], 100, alpha=0.1
        ),
        verbose=0,
    )

    if RECORD_CKPTS:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=(output_dir + CKPT_DIR),
            save_weights_only=True,
            save_best_only=False,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        )
    else:
        ckpt_callback = None

    try:
        model.fit(
            ds_train,
            epochs=options["epochs"],
            validation_data=ds_val,
            callbacks=[tb_callback, reservoir_cb, ckpt_callback, lr_callback],
            verbose=1,
        )
    except Exception as e:
        print(e)

    print("\nRUNNING EVALUATION OVER TEST SET\n")
    try:
        model.evaluate(
            ds_test,
            verbose=1,
        )
    except Exception as e:
        print(e)

    return output_dir + CKPT_DIR
