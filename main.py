import time
import threading
from mnist_rnn_model import get_model
from utils.model_utils import mod_sign_with_tanh_deriv, sign_with_tanh_deriv
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import io
import os
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit("autoclustering")

tf.random.set_seed(1997)  # For experimental reproducibility

RUNS_DIR = "runs/"
TB_LOGS_DIR = "logs/tensorboard/"
CKPT_DIR = "checkpoints/"
BACKUP_DIR = "tmp/backup"
RECORD_CKPTS = True


def configure_environment():
    # Set dtype
    dtype = tf.float32

    # Get GPU info
    gpus = tf.config.list_physical_devices('GPU')

    # Init gpus
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(str(e))

        device = gpus[0]
        print('Running single gpu: {}'.format(device))
        strategy = tf.distribute.OneDeviceStrategy(
            device=device)

    else:
        device = tf.config.list_physical_devices('CPU')[0]
        print('Running on CPU: {}'.format(device))
        strategy = tf.distribute.OneDeviceStrategy(
            device=device)

    return strategy, dtype


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


def resize(image, label):
    return tf.image.resize(image, [128, 128]), label


def augment(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)

    return image, label


def get_datasets(batch_size: int = 512) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    ds, ds_test = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=False,
        as_supervised=True,
        with_info=False,
    )
    ds_val = ds.take(2500)
    ds_train = ds.skip(2500)

    autotune = tf.data.experimental.AUTOTUNE

    # Training dataset
    ds_train = ds_train.map(normalize_img, num_parallel_calls=autotune)
    if options["enlarge"]:
        ds_train = ds_train.map(resize, num_parallel_calls=autotune)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle()
    ds_train = ds_train.map(augment, num_parallel_calls=autotune)
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(autotune)

    # Validation dataset
    ds_val = ds_val.map(normalize_img, num_parallel_calls=autotune)
    if options["enlarge"]:
        ds_val = ds_val.map(resize, num_parallel_calls=autotune)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(autotune)

    # Test dataset
    ds_test = ds_test.map(normalize_img, num_parallel_calls=autotune)
    if options["enlarge"]:
        ds_test = ds_test.map(resize, num_parallel_calls=autotune)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(autotune)

    return ds_train, ds_val, ds_test


def train(pretrained_weights: str | None, options, layer_options) -> str:
    strategy, _, _ = configure_environment()

    now = datetime.now()
    RUN_DIR = RUNS_DIR + \
        now.strftime("%Y%m") + "/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    BATCHSIZE = options["batch_size"]
    ds_train, ds_val, ds_test = get_datasets(BATCHSIZE)

    with strategy.scope():
        model = get_model(options, layer_options)

        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)
            print('Restored pretrained weights from {}.'.format(pretrained_weights))

        # Reset the stat variables
        weights = model.get_weights()
        for i in range(len(weights)):
            if "/w" in model.weights[i].name or "/x" in model.weights[i].name:
                weights[i] = 0*weights[i]
        model.set_weights(weights)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=options["learning_rate"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False),
            metrics=["accuracy"],
        )

    # model.run_eagerly = True

    # Define callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=(RUN_DIR + TB_LOGS_DIR), histogram_freq=1, update_freq="epoch",
    )

    # Add a lr decay callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        tf.keras.optimizers.schedules.CosineDecay(
            options["learning_rate"], 100, alpha=0.1),
        verbose=0,
    )

    if RECORD_CKPTS:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(RUN_DIR + CKPT_DIR),
                                                           save_weights_only=True,
                                                           save_best_only=False,
                                                           monitor="val_accuracy",
                                                           mode="max",
                                                           verbose=1)
    else:
        ckpt_callback = None

    train = True
    test = True

    if train:
        try:
            model.fit(
                ds_train,
                epochs=1000,
                validation_data=ds_val,
                callbacks=[tb_callback, ckpt_callback, lr_callback],
                verbose=1,
            )
        except Exception as e:
            print(e)

    if test:
        model.evaluate(
            x=ds_test,
            verbose=1,
        )

    return RUN_DIR


def get_model_parameter_stats(pretrained_weights: str, options, layer_options):
    tern_params = {
        "QRNN_0": 0,
        "QRNN_1": 0,
        "DENSE_0": 0,
        "DENSE_OUT": 0
    }
    model = get_model(options, layer_options)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
        print('Restored pretrained weights from {}.'.format(pretrained_weights))
    else:
        return tern_params

    # Calculate weight stats
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:
            all_weights = tf.concat(
                [tf.reshape(x, shape=[-1]) for x in layer.trainable_weights], axis=-1)
            mean_abs = tf.math.reduce_mean(tf.abs(all_weights))
            print(layer.name)
            if (layer.name.find("QRNN_0") != -1):
                tern_params["QRNN_0"] = mean_abs
            if (layer.name.find("QRNN_1") != -1):
                tern_params["QRNN_1"] = mean_abs
            if (layer.name.find("DENSE_0") != -1):
                tern_params["DENSE_0"] = mean_abs
            if (layer.name.find("DENSE_OUT") != -1):
                tern_params["DENSE_OUT"] = mean_abs

    return tern_params


# RUN THE MODIFIED FOUR-STEP QUANTIZATION PROCEDURE
pretrained_weights = None
options = {
    "enlarge": False,
    "learning_rate": 1e-4,
    "batch_size": 512,
    "t": 1.5,
    "tᵢ": 0.7,
    "s": 4.0,
    "oar": {
        "lm": 1e-4,
        "precision": 6,
    },
    "quantize": False


}

for step in range(1, 5):
    # Change settings according to step number
    ternarize_inputs = False
    t = 1.0
    s = 1.0
    activation = tf.keras.activations.tanh
    oar = False
    tern_params = {
        "QRNN_0": 0,
        "QRNN_1": 0,
        "DENSE_0": 0,
        "DENSE_OUT": 0
    }
    if step == 2:
        s = options["s"]
        activation = sign_with_tanh_deriv
    if step == 3:
        ternarize_inputs = True
    if step == 4:
        options["quantize"] = True
        t = options["t"]
        oar = True
        tern_params = get_model_parameter_stats(
            pretrained_weights, options, layer_options)

        def activation(x): return mod_sign_with_tanh_deriv(
            x, num_bits=options["oar"]["precision"])

    # Adjust layer options
    layer_options = {
        "INPUT": {
            "ternarize": ternarize_inputs
        },
        "QRNN_0": {
            "activation": activation,
            "oar": {
                "use": oar,
                "lm": options["oar"]["lm"],
                "precision": options["oar"]["precision"],
            },
            "s": s,
            "τ": t*tern_params["QRNN_0"],
        },
        "QRNN_1": {
            "activation": activation,
            "oar": {
                "use": oar,
                "lm": options["oar"]["lm"],
                "precision": options["oar"]["precision"],
            },
            "s": s,
            "τ": t*tern_params["QRNN_1"],
        },
        "DENSE_0": {
            "activation": activation,
            "oar": {
                "use": oar,
                "lm": options["oar"]["lm"],
                "precision": options["oar"]["precision"],
            },
            "s": 1.0,
            "τ": t*tern_params["DENSE_0"],
        },
        "DENSE_OUT": {
            "activation": lambda x: tf.keras.activations.softmax(x),
            "oar": {
                "use": True,
                "lm": 0,
                "precision": options["oar"]["precision"],
            },
            "s": 1.0,
            "τ": t*tern_params["DENSE_OUT"],
        }
    }

    pretrained_weights = train(pretrained_weights=pretrained_weights,
                               options=options, layer_options=layer_options)


# lrs = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024]
# oar_lms = [0, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]

# threads = []
# for lr in lrs:
#     for oar_lm in oar_lms:
#         oar_lm_real = 0.001 * oar_lm

#         options["oar"]["lm"] = oar_lm_real
#         layer_options["QRNN_0"]["oar"]["lm"] = oar_lm_real
#         layer_options["QRNN_1"]["oar"]["lm"] = oar_lm_real
#         layer_options["DENSE_0"]["oar"]["lm"] = oar_lm_real
#         layer_options["DENSE_OUT"]["oar"]["lm"] = oar_lm_real

#         print("LR = %f, OAR_LM = %f" % (lr, oar_lm))
#         thr = threading.Thread(target=run_one_train, args=(options, layer_options, 0.001*lr,))
#         threads.append(thr)
#         thr.start()

#         time.sleep(30)
#         # run_one_train(options, layer_options, 0.01*lr)
#     for index, thread in enumerate(threads):
#         print("Main    : before joining thread %d." % index)
#         thread.join()
#         print("Main    : thread %d done" % index)

print("End!")
