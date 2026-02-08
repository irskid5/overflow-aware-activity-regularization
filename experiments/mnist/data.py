"""MNIST dataset loading and preprocessing."""

import tensorflow as tf
import tensorflow_datasets as tfds


def normalize_img(image, label):
    """Normalizes the MNIST image by dividing by the max value.

    Args:
        image: MNIST image tensor
        label: MNIST image label

    Returns:
        Tuple of (normalized image, label)
    """
    return tf.cast(image, tf.float32) / 255.0, label


def resize(image, label):
    """Resizes the image to [128, 128].

    Args:
        image: Image tensor
        label: Label tensor

    Returns:
        Tuple of (resized image, label)
    """
    return tf.image.resize(image, [128, 128]), label


def augment(image, label):
    """Applies data augmentation: random brightness and horizontal flip.

    Args:
        image: Image tensor
        label: Label tensor

    Returns:
        Tuple of (augmented image, label)
    """
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)
    return image, label


def _prepare_dataset(
    ds: tf.data.Dataset,
    enlarge: bool,
    batch_size: int,
    shuffle_buffer: int | None = None,
    apply_augment: bool = False,
) -> tf.data.Dataset:
    """Applies preprocessing pipeline to a dataset.

    Args:
        ds: Input dataset
        enlarge: If True, resize images to 128x128
        batch_size: Batch size
        shuffle_buffer: If set, shuffle with this buffer size
        apply_augment: If True, apply data augmentation

    Returns:
        Preprocessed dataset
    """
    autotune = tf.data.experimental.AUTOTUNE
    ds = ds.map(normalize_img, num_parallel_calls=autotune)
    if enlarge:
        ds = ds.map(resize, num_parallel_calls=autotune)
    ds = ds.cache()
    if shuffle_buffer is not None:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    if apply_augment:
        ds = ds.map(augment, num_parallel_calls=autotune)
    return ds.batch(batch_size, drop_remainder=True).prefetch(autotune)


def get_datasets(
    batch_size: int = 512, enlarge: bool = False
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Downloads, preprocesses, and prepares the MNIST dataset.

    Creates three dataloaders: training, validation, and test.
    - Validation set: 2500 samples from training data
    - Training set: 57500 samples
    - Test set: 10000 samples

    Optionally resizes from [28,28] to [128,128] for enlarged MNIST RNN.

    Args:
        batch_size: Batch size for training. Defaults to 512.
        enlarge: If True, resize images to 128x128. Defaults to False.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
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

    ds_train = _prepare_dataset(
        ds_train, enlarge, batch_size,
        shuffle_buffer=len(ds_train), apply_augment=True
    )
    ds_val = _prepare_dataset(ds_val, enlarge, batch_size)
    ds_test = _prepare_dataset(ds_test, enlarge, batch_size)

    return ds_train, ds_val, ds_test
