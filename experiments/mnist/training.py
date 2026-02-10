"""MNIST training utilities."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import datetime

import tensorflow as tf

RUNS_DIR = "runs/mnist/"


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
