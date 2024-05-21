from mnist_rnn_model import get_model
from utils.model_utils import (
    mod_sign_with_tanh_deriv,
    sign_with_tanh_deriv,
    get_default_layer_options_from_options,
)
from export_mnist_weights_h5 import export_mnist_weights
from export_mnist import extract_ternarized_mnist_test_dataset
import tensorflow_datasets as tfds
import tensorflow as tf
import os
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit("autoclustering")

tf.random.set_seed(1997)  # For experimental reproducibility

RUNS_DIR = "runs/"
TB_LOGS_DIR = "logs/tensorboard/"
CKPT_DIR = "checkpoints/"
BACKUP_DIR = "tmp/backup"
RECORD_CKPTS = True


def configure_environment():
    """Configures the environment by selecting the datatype and device strategy."""
    # Set dtype
    dtype = tf.float32

    # Get GPU info
    gpus = tf.config.list_physical_devices("GPU")

    # Init gpus
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


def normalize_img(image, label):
    """Normalizes the MNIST image by dividing by the max value.

    Args:
        image (_type_): mnist image
        label (_type_): mnist image label

    Returns:
        _type_: normalized image, label
    """
    return tf.cast(image, tf.float32) / 255.0, label


def resize(image, label):
    """Resizes the image to [128,128]"""
    return tf.image.resize(image, [128, 128]), label


def augment(image, label):
    """Changes brightness levels randomly and flips images left or right randomly."""
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)

    return image, label


def get_datasets(
    batch_size: int = 512, enlarge: bool = False
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Downloads, preprocesses, and prepares the MNIST dataset into three dataloaders:
    training, validation, and test. The validation set is 2500 samples, from the training
    dataset. The training dataset contains 47500 samples, and the test dataset contains
    10000 samples. Resizes the dataset from [28,28] to [128,128] for enlarged MNIST RNN
    run, if selected.

    Args:
        batch_size (int, optional): the batch size used for training. Defaults to 512.
        enlarge (bool, optional): resize for enlarged MNIST RNN or not. Defaults to False.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: training, validation,
        and test datasets
    """
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
    if enlarge:
        ds_train = ds_train.map(resize, num_parallel_calls=autotune)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size=len(ds_train))
    ds_train = ds_train.map(augment, num_parallel_calls=autotune)
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(autotune)

    # Validation dataset
    ds_val = ds_val.map(normalize_img, num_parallel_calls=autotune)
    if enlarge:
        ds_val = ds_val.map(resize, num_parallel_calls=autotune)
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(batch_size, drop_remainder=True)
    ds_val = ds_val.prefetch(autotune)

    # Test dataset
    ds_test = ds_test.map(normalize_img, num_parallel_calls=autotune)
    if enlarge:
        ds_test = ds_test.map(resize, num_parallel_calls=autotune)
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size, drop_remainder=True)
    ds_test = ds_test.prefetch(autotune)

    return ds_train, ds_val, ds_test


def train(pretrained_weights: str | None, options, layer_options) -> str:
    """Runs training for a number of epochs, set in options. Loads the pretrained weights,
    denoted by pretrained_weights, gets the model according to layer_options/options, and
    runs the training based on options. Also runs an evaluation over the test set at the
    end of the training. Prints everything to console. Returns the pretrained weights folder
    for the current training run.

    Args:
        pretrained_weights (str | None): path to checkpoints folder containing parameters to initialize from.
        options (_type_): Set of options, including learning rate, batch size, ternarization and gradient scales
        (from four-step quantization algorithm).
        layer_options (_type_): Set of options for each layer of the MNIST RNN, including ternarization treshold, etc.

    Returns:
        str: path to checkpoints folder of trained model
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

        # Reset the stat variables
        weights = model.get_weights()
        for i in range(len(weights)):
            if (
                "/w" in model.weights[i].name
                or "/x" in model.weights[i].name
                or "preacts" in model.weights[i].name
            ):
                weights[i] = 0 * weights[i]
        model.set_weights(weights)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=options["learning_rate"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    # model.run_eagerly = True

    # TensorBoard callback.
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=(RUN_DIR + TB_LOGS_DIR),
        histogram_freq=1,
        update_freq="epoch",
    )

    # Add a learning rate schedule
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

    train = True
    test = True

    if train:
        try:
            model.fit(
                ds_train,
                epochs=options["epochs"],
                validation_data=ds_val,
                callbacks=[tb_callback, ckpt_callback, lr_callback],
                verbose=1,
            )
        except Exception as e:
            print(e)

    if test:
        print("\nRUNNING EVALUATION OVER TEST SET\n")
        try:
            model.evaluate(
                ds_test,
                verbose=1,
            )
        except Exception as e:
            print(e)

    return RUN_DIR + CKPT_DIR


def get_model_parameter_stats(pretrained_weights: str, options, layer_options):
    """Gets the mean of the absolute value for ternarization of each parameter, per layer.

    Args:
        pretrained_weights (str): path to checkpoints folder containing parameters
        options (_type_): model options
        layer_options (_type_): layer-wise model options

    Returns:
        _type_: _description_
    """
    tern_params = {"QRNN_0": 0, "QRNN_1": 0, "DENSE_0": 0, "DENSE_OUT": 0}
    model = get_model(options, layer_options)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
        print("Restored pretrained weights from {}.".format(pretrained_weights))
    else:
        return tern_params

    # Calculate weight stats
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:
            all_weights = tf.concat(
                [tf.reshape(x, shape=[-1]) for x in layer.trainable_weights], axis=-1
            )
            mean_abs = tf.math.reduce_mean(tf.abs(all_weights))
            if layer.name.find("QRNN_0") != -1:
                tern_params["QRNN_0"] = mean_abs.numpy()
            if layer.name.find("QRNN_1") != -1:
                tern_params["QRNN_1"] = mean_abs.numpy()
            if layer.name.find("DENSE_0") != -1:
                tern_params["DENSE_0"] = mean_abs.numpy()
            if layer.name.find("DENSE_OUT") != -1:
                tern_params["DENSE_OUT"] = mean_abs.numpy()
    return tern_params


def perform_step_in_four_step_quant(step: int, pretrained_weights: str, options) -> str:
    """Performs a step in the modified four-step quantization procedure.

    Args:
        step (int): step number (1 → 4)
        pretrained_weights (str): path to checkpoints folder the weights to initialize the step with.
        options (_type_): overall options, including ternarization scale t and gradient scale s.

    Returns:
        str: path to checkpoints folder of the trained model from the step
    """
    print(f"\nPERFORMING STEP {step}/4 FROM FOUR-STEP QUANTIZATION PROCESS\n")

    layer_options = get_default_layer_options_from_options(options)

    # Change settings according to step number
    ternarize_inputs = False
    t = 1.0
    s = 1.0
    activation = tf.keras.activations.tanh
    oar = False
    tern_params = {"QRNN_0": 0, "QRNN_1": 0, "DENSE_0": 0, "DENSE_OUT": 0}
    if step == 2:
        s = options["s"]
        activation = sign_with_tanh_deriv
    if step == 3:
        ternarize_inputs = True
    if step == 4:
        options["quantize"] = True
        options["learning_rate"] *= 0.1
        t = options["t"]
        oar = True
        tern_params = get_model_parameter_stats(
            pretrained_weights, options, layer_options
        )
        print("\nTERNARIZATION PARAMETERS:")
        print(tern_params)

        def activation(x):
            return mod_sign_with_tanh_deriv(x, num_bits=options["oar"]["precision"])

    # Adjust layer options
    layer_options = {
        "INPUT": {"ternarize": ternarize_inputs},
        "QRNN_0": {
            "activation": activation,
            "oar": {
                "use": oar,
                "lm": options["oar"]["lm"],
                "precision": options["oar"]["precision"],
            },
            "s": s,
            "τ": t * tern_params["QRNN_0"],
        },
        "QRNN_1": {
            "activation": activation,
            "oar": {
                "use": oar,
                "lm": options["oar"]["lm"],
                "precision": options["oar"]["precision"],
            },
            "s": s,
            "τ": t * tern_params["QRNN_1"],
        },
        "DENSE_0": {
            "activation": activation,
            "oar": {
                "use": oar,
                "lm": options["oar"]["lm"],
                "precision": options["oar"]["precision"],
            },
            "s": 1.0,
            "τ": t * tern_params["DENSE_0"],
        },
        "DENSE_OUT": {
            "activation": lambda x: tf.keras.activations.softmax(x),
            "oar": {
                "use": True,
                "lm": 0.0,
                "precision": options["oar"]["precision"],
            },
            "s": 1.0,
            "τ": t * tern_params["DENSE_OUT"],
        },
    }

    print("OPTIONS AND LAYER OPTIONS:")
    print(options)
    print(layer_options)

    return train(
        pretrained_weights=pretrained_weights,
        options=options,
        layer_options=layer_options,
    )


def perform_four_step_quant(options) -> str:
    """Performs the four-step quantization procedure. The initial model is the MNIST RNN, without pretrained weights.

    Args:
        options (_type_): _description_

    Returns:
        str: path to checkpoint folder of final step parameters
    """
    pretrained_weights = None
    for step in range(1, 5):
        pretrained_weights = perform_step_in_four_step_quant(
            step=step, pretrained_weights=pretrained_weights, options=options
        )
    return pretrained_weights


#########################################################################################################
# RUN THE MODIFIED FOUR-STEP QUANTIZATION PROCEDURE FOR THE FOLLOWING OPTIONS


def train_quantize_extract_MNIST_RNN() -> str:
    """This code takes the options below, runs the four-step quantization procedure and trains a quantized model.
    It then extracts the weights. It returns the pretrained weights from the last step.
    """
    options = {
        "enlarge": False,
        "epochs": 1,
        "learning_rate": 1e-4,
        "batch_size": 512,
        "t": 1.5,
        "tᵢ": 0.7,
        "s": 4.0,
        "oar": {
            "lm": 1e-4,
            "precision": 6,
        },
        "quantize": False,
    }
    final_parameters = perform_four_step_quant(options)
    export_mnist_weights(final_parameters)
    return final_parameters


def train_quantize_extract_enlarged_MNIST_RNN() -> str:
    """This code takes the options below, runs the four-step quantization procedure and trains a quantized model.
    It then extracts the weights. It returns the pretrained weights from the last step.
    """
    options = {
        "enlarge": True,
        "epochs": 1,
        "learning_rate": 5e-6,
        "batch_size": 512,
        "t": 1.5,
        "tᵢ": 0.7,
        "s": 4.0,
        "oar": {
            "lm": 1e-4,
            "precision": 6,
        },
        "quantize": False,
    }
    final_parameters = perform_four_step_quant(options)
    export_mnist_weights(final_parameters)
    return final_parameters


def evaluation_with_and_without_oar2():
    # First get third step model
    third_step_options = {
        "enlarge": False,
        "epochs": 1,
        "learning_rate": 1e-4,
        "batch_size": 512,
        "t": 1.5,
        "tᵢ": 0.7,
        "s": 4.0,
        "oar": {
            "lm": 1e-4,
            "precision": 6,
        },
        "quantize": False,
    }
    third_step = None
    for step in range(1, 4):
        third_step = perform_step_in_four_step_quant(
            step=step, pretrained_weights=third_step, options=third_step_options
        )
    for i in range(2):
        for ω in range(3, 9):
            cur_options = third_step_options.copy()
            cur_options["oar"]["precision"] = ω
            cur_options["oar"]["lm"] = i * 1e-3
            perform_step_in_four_step_quant(
                step=4, pretrained_weights=third_step, options=cur_options
            )


def evaluation_different_oar_regularization_rates():
    # First get third step model
    third_step_options = {
        "enlarge": False,
        "epochs": 1,
        "learning_rate": 1e-4,
        "batch_size": 512,
        "t": 1.5,
        "tᵢ": 0.7,
        "s": 4.0,
        "oar": {
            "lm": 1e-4,
            "precision": 6,
        },
        "quantize": False,
    }
    third_step = None
    for step in range(1, 4):
        third_step = perform_step_in_four_step_quant(
            step=step, pretrained_weights=third_step, options=third_step_options
        )
    rates = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    bits = [5, 6]
    for ω in bits:
        for lm in rates:
            cur_options = third_step_options.copy()
            cur_options["oar"]["precision"] = ω
            cur_options["oar"]["lm"] = lm
            perform_step_in_four_step_quant(
                step=4, pretrained_weights=third_step, options=cur_options
            )


if __name__ == "__main__":
    train_quantize_extract_MNIST_RNN()
    train_quantize_extract_enlarged_MNIST_RNN()
    evaluation_with_and_without_oar2()
    evaluation_different_oar_regularization_rates()
    extract_ternarized_mnist_test_dataset()
    print("End!")
