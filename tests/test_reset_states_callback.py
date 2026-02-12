"""Tests for ResetStatesCallback."""
import pytest
import tensorflow as tf


def test_callback_resets_states_on_epoch_begin():
    """Callback calls model.reset_states() at epoch start."""
    from oar.callbacks import ResetStatesCallback

    # Create a simple stateful model
    # RNN expects 3D input: (batch, timesteps, features)
    inputs = tf.keras.Input(batch_shape=(2, 5, 1))
    x = tf.keras.layers.SimpleRNN(4, stateful=True, return_sequences=True)(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    # Run one batch to set states
    x_data = tf.random.uniform((2, 5, 1), maxval=1.0)
    y_data = tf.random.uniform((2, 5), maxval=3, dtype=tf.int32)
    model.train_on_batch(x_data, y_data)

    # Get state after training
    rnn_layer = model.layers[1]
    state_before = rnn_layer.states[0].numpy().copy()

    # Trigger epoch begin callback
    callback = ResetStatesCallback()
    callback.set_model(model)
    callback.on_epoch_begin(epoch=0)

    # State should be reset (all zeros)
    state_after = rnn_layer.states[0].numpy()
    assert (state_after == 0).all()
    assert not (state_before == state_after).all()


def test_callback_works_with_model_fit():
    """Callback integrates with model.fit()."""
    from oar.callbacks import ResetStatesCallback

    # Create stateful model
    # RNN expects 3D input: (batch, timesteps, features)
    inputs = tf.keras.Input(batch_shape=(2, 5, 1))
    x = tf.keras.layers.SimpleRNN(4, stateful=True, return_sequences=True)(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    # Create small dataset
    x_data = tf.random.uniform((4, 5, 1), maxval=1.0)
    y_data = tf.random.uniform((4, 5), maxval=3, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(2)

    # Train with callback - should not raise
    callback = ResetStatesCallback()
    history = model.fit(dataset, epochs=2, callbacks=[callback], verbose=0)
    
    assert len(history.history["loss"]) == 2
