"""MNIST training utilities."""

import os
from datetime import datetime

import tensorflow as tf

from oar import ReservoirHistogramCallback, reset_stat_weights
from experiments.mnist.data import get_datasets
from experiments.mnist.model import get_model

RUNS_DIR = "runs/"
TB_LOGS_DIR = "logs/tensorboard/"
CKPT_DIR = "checkpoints/"
RECORD_CKPTS = True


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


def train(pretrained_weights: str | None, options, layer_options) -> str:
    """Runs training for a number of epochs.

    Loads pretrained weights, builds model, and runs training.
    Also runs evaluation over test set at the end.

    Args:
        pretrained_weights: Path to checkpoints folder for initialization
        options: Training options dict
        layer_options: Per-layer options dict

    Returns:
        Path to checkpoints folder of trained model
    """
    strategy, _ = configure_environment()

    now = datetime.now()
    RUN_DIR = (
        RUNS_DIR + now.strftime("%Y%m") + "/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    )

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
        log_dir=(RUN_DIR + TB_LOGS_DIR),
        histogram_freq=1,
        update_freq="epoch",
    )
    reservoir_cb = ReservoirHistogramCallback(log_dir=(RUN_DIR + TB_LOGS_DIR))

    # Learning rate schedule
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        tf.keras.optimizers.schedules.CosineDecay(
            options["learning_rate"], 100, alpha=0.1
        ),
        verbose=0,
    )

    if RECORD_CKPTS:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=(RUN_DIR + CKPT_DIR),
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

    return RUN_DIR + CKPT_DIR
