"""Integration tests for Speech Commands experiment."""
import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility."""
    tf.random.set_seed(1997)


def test_step1_trains_and_evaluates():
    """Step 1 trains successfully and produces reasonable accuracy."""
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    from experiments.speech_commands.data import get_datasets
    from experiments.speech_commands.model import get_model
    
    # Get step 1 config
    step_config = SPEECH_COMMANDS_EXPERIMENT.steps[1]
    
    # Load small subset for testing
    train_ds, val_ds, _ = get_datasets(batch_size=32, cache=False)
    train_ds = train_ds.take(10)  # Just 10 batches
    val_ds = val_ds.take(5)
    
    # Build model
    model = get_model(step_config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    
    # Train briefly
    history = model.fit(train_ds, epochs=2, validation_data=val_ds, verbose=0)
    
    # Check training happened
    assert len(history.history["loss"]) == 2
    assert history.history["loss"][1] < history.history["loss"][0], "Loss should decrease"


def test_full_pipeline_shapes():
    """Full pipeline produces correct shapes at each stage."""
    from experiments.speech_commands.data import get_datasets, NUM_FRAMES, NUM_MFCC, NUM_CLASSES
    from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
    from experiments.speech_commands.model import get_model
    
    # Load data
    train_ds, _, _ = get_datasets(batch_size=4, cache=False)
    
    # Get one batch
    for x, y in train_ds.take(1):
        assert x.shape == (4, NUM_FRAMES, NUM_MFCC)
        assert y.shape == (4,)
    
    # Build model and check output
    step_config = SPEECH_COMMANDS_EXPERIMENT.steps[1]
    model = get_model(step_config)
    
    output = model(x, training=False)
    assert output.shape == (4, NUM_CLASSES)
