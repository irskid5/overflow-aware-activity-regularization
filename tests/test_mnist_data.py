"""Tests for experiments/mnist/data.py."""

import tensorflow as tf


def test_normalize_img_divides_by_255():
    from experiments.mnist.data import normalize_img

    image = tf.constant([[[255.0]]], dtype=tf.float32)
    label = tf.constant(5)
    norm_img, norm_label = normalize_img(image, label)
    tf.debugging.assert_near(norm_img, tf.constant([[[1.0]]]))
    tf.debugging.assert_equal(norm_label, label)


def test_resize_to_128x128():
    from experiments.mnist.data import resize

    image = tf.zeros([28, 28, 1], dtype=tf.float32)
    label = tf.constant(3)
    resized_img, resized_label = resize(image, label)
    assert resized_img.shape == (128, 128, 1)
    tf.debugging.assert_equal(resized_label, label)


def test_get_datasets_returns_three_datasets():
    from experiments.mnist.data import get_datasets

    ds_train, ds_val, ds_test = get_datasets(batch_size=32, enlarge=False)
    assert isinstance(ds_train, tf.data.Dataset)
    assert isinstance(ds_val, tf.data.Dataset)
    assert isinstance(ds_test, tf.data.Dataset)


def test_get_datasets_batch_shape_28x28():
    from experiments.mnist.data import get_datasets

    ds_train, _, _ = get_datasets(batch_size=32, enlarge=False)
    for batch in ds_train.take(1):
        images, labels = batch
        assert images.shape == (32, 28, 28, 1)
        assert labels.shape == (32,)


def test_get_datasets_batch_shape_128x128():
    from experiments.mnist.data import get_datasets

    ds_train, _, _ = get_datasets(batch_size=32, enlarge=True)
    for batch in ds_train.take(1):
        images, labels = batch
        assert images.shape == (32, 128, 128, 1)


def test_augment_returns_same_shape():
    from experiments.mnist.data import augment

    image = tf.random.uniform([28, 28, 1], dtype=tf.float32)
    label = tf.constant(7)
    aug_image, aug_label = augment(image, label)
    assert aug_image.shape == image.shape
    tf.debugging.assert_equal(aug_label, label)


def test_normalize_img_preserves_zero():
    from experiments.mnist.data import normalize_img

    image = tf.zeros([28, 28, 1], dtype=tf.float32)
    label = tf.constant(0)
    norm_image, _ = normalize_img(image, label)
    tf.debugging.assert_near(norm_image, tf.zeros_like(norm_image))
