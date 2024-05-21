"""
This file prepares the MNIST dataset test set for TFHE inference using Concrete Core + Concrete GPU.
Since the libraries are written in Rust, we want Rust to contain arrays of data the same way
TF does. 

This file preprocesses the test dataset, normalizes the vectors and saves two .npy + one .npz file 
that zips together two arrays:
    1) x: matrix of [n, 28, 28] ternary values {-1, 0, 1}
    2) y_true: matrix of [n, 1] integers in [0, 9] depicting the 10 class labels

The "x" matrix is the batch of 28x28 pixel (normalized, ternarized) test images.

The ternarization is defined by a threshold theta = 0.7*tf.reduce_mean(tf.abs(batch))) as in
Ternary Weight Networks (TWN).
"""

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from quantization import ternarize_tensor_with_threshold


def cast_tensor_to_float32(tensor):
    return tf.cast(tensor, tf.float32)


def normalize_img(image):
    """Normalizes images. Image is float32."""
    return image / 255.0


def ternarize_img(image):
    return ternarize_tensor_with_threshold(
        image, theta=0.7 * tf.reduce_mean(tf.abs(image))
    )


def cast_tensor_to_int8(tensor):
    return tf.cast(tensor, tf.int8)


def reshape_img(image):
    return tf.squeeze(image, axis=-1)


def resize_img(image):
    return tf.image.resize(image, [128, 128])


def prepare_for_tfhe_enlarged(image, label):
    img = resize_img(image)
    img = cast_tensor_to_float32(img)
    img = normalize_img(img)
    img = ternarize_img(img)
    img = cast_tensor_to_int8(img)
    lbl = cast_tensor_to_int8(label)
    return img, lbl


def prepare_for_tfhe_regular(image, label):
    img = reshape_img(image)
    img = cast_tensor_to_float32(img)
    img = normalize_img(img)
    img = ternarize_img(img)
    img = cast_tensor_to_int8(img)
    lbl = cast_tensor_to_int8(label)
    return img, lbl


def extract_ternarized_mnist_test_dataset():
    # Load MNIST using tfds
    ds_test, ds_info = tfds.load(
        "mnist",
        split="test",
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    print("Loaded MNIST test set as supervised")
    for enlarge in [True, False]:
        # Preprocess the data
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds_test = ds_test.map(
            prepare_for_tfhe_enlarged if enlarge else prepare_for_tfhe_regular,
            num_parallel_calls=AUTOTUNE,
        )

        print("Preprocessed MNIST for TFHE")

        # Iterate over the dataset and append the elements to the list
        image_list = []
        label_list = []
        for example in ds_test:
            image_list.append(example[0].numpy())
            label_list.append(example[1].numpy())
        image_list, label_list = np.array(image_list), np.array(label_list)

        print("Converted to Numpy")

        # Save files
        directory = "./mnist_preprocessed"
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(
            directory + f'/mnist_images_norm_tern{"_128x128" if enlarge else ""}.npy',
            image_list,
        )
        np.save(
            directory + f'/mnist_labels{"_128x128" if enlarge else ""}.npy', label_list
        )
        np.savez(
            directory + f'/mnist_norm_tern{"_128x128" if enlarge else ""}.npz',
            X=image_list,
            y=label_list,
        )

    print("Saved NPY+NPZ files of preprocessed MNIST Images and Labels")
