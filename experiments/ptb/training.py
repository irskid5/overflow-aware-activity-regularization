"""PTB training utilities.

Provides adapters to use existing oar.runner.Runner with PTB:
- make_data_loader(): Creates DataLoader-compatible callable
- run_ptb(): Convenience function for running PTB experiment
- PTBRunner: Extends Runner with ResetStatesCallback for stateful RNNs

Key design: Uses ResetStatesCallback with Runner instead of
creating entirely new training logic.
"""
import os
from typing import Callable

import tensorflow as tf

from experiments.ptb.config import BATCH_SIZE, NUM_STEPS
from experiments.ptb.data import get_datasets
from experiments.ptb.model import make_model_factory
from oar import Perplexity, ResetStatesCallback, ReservoirHistogramCallback, reset_stat_weights
from oar.config import ExperimentConfig, StepConfig
from oar.runner import Runner


def make_data_loader() -> Callable:
    """Create DataLoader-compatible callable for PTB.
    
    The returned callable matches oar.runner.DataLoader protocol:
    (batch_size, enlarge) -> (train_ds, val_ds, test_ds)
    
    PTB ignores the `enlarge` parameter (not applicable).
    
    Returns:
        DataLoader-compatible callable
    """
    def data_loader(batch_size: int, enlarge: bool):
        # Load data - PTB ignores enlarge
        train_ds, val_ds, test_ds, _ = get_datasets(batch_size, NUM_STEPS)
        return train_ds, val_ds, test_ds
    
    return data_loader


def get_vocab_size() -> int:
    """Get PTB vocabulary size (loads data if needed).
    
    Returns:
        Vocabulary size (~10000)
    """
    _, _, _, vocab_size = get_datasets(BATCH_SIZE, NUM_STEPS)
    return vocab_size


def evaluate_perplexity(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> float:
    """Evaluate perplexity on dataset.

    Args:
        model: Compiled model
        dataset: Evaluation dataset

    Returns:
        Perplexity (exp of average cross-entropy)
    """
    model.reset_states()
    metric = Perplexity()

    for x, y in dataset:
        y_pred = model(x, training=False)
        metric.update_state(y, y_pred)

    return float(metric.result().numpy())


class PTBRunner(Runner):
    """Runner for PTB with ResetStatesCallback for stateful RNNs.
    
    Extends the base Runner to add ResetStatesCallback, which resets
    RNN hidden states at epoch boundaries. This is required for language
    modeling where states persist across batches within an epoch.
    """

    def _run_step(
        self,
        step_num: int,
        step_config: StepConfig,
        pretrained_weights: str | None,
        run_dir: str,
    ) -> str:
        """Execute a single training step with ResetStatesCallback.
        
        Same as parent but adds ResetStatesCallback for stateful RNNs.
        """
        output_dir = os.path.join(run_dir, f"step_{step_num}")
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        ds_train, ds_val, ds_test = self.data_loader(
            batch_size=step_config.batch_size,
            enlarge=step_config.enlarge,
        )

        # Build model (model_factory computes thresholds internally if needed)
        model = self.model_factory(step_config, pretrained_weights=pretrained_weights)

        reset_stat_weights(model)

        # Compile with initial LR (decay handled by callback)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=step_config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        
        # LR schedule decays based on epochs (not optimizer steps)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=step_config.learning_rate,
            decay_steps=step_config.cosine_decay_epochs,
            alpha=step_config.cosine_decay_alpha,
        )
        lr_callback = tf.keras.callbacks.LearningRateScheduler(
            lr_schedule, verbose=0
        )

        callbacks = [
            lr_callback,
            ResetStatesCallback(),  # PTB-specific: reset states at epoch boundaries
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(output_dir, "logs/tensorboard")
            ),
            ReservoirHistogramCallback(
                log_dir=os.path.join(output_dir, "logs/tensorboard")
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, "checkpoints/"),
                save_weights_only=True,
            ),
        ]

        model.fit(
            ds_train,
            epochs=step_config.epochs,
            validation_data=ds_val,
            callbacks=callbacks,
            verbose=2,
        )

        print("\nEvaluating on test set...")
        model.reset_states()  # Reset before evaluation
        model.evaluate(ds_test, verbose=2)

        checkpoint_path = os.path.join(output_dir, "checkpoints/")
        self._save_step_config(output_dir, step_num, step_config, checkpoint_path)

        return checkpoint_path


def run_ptb(
    experiment: ExperimentConfig,
    start_step: int = 1,
    end_step: int = 4,
    resume_from: str | None = None,
) -> str:
    """Run PTB experiment using PTBRunner with ResetStatesCallback.
    
    This is a convenience function that:
    1. Gets vocab_size from data
    2. Creates model_factory with vocab_size bound
    3. Creates data_loader adapter
    4. Runs with PTBRunner
    
    Args:
        experiment: PTB experiment configuration
        start_step: First step to run (1-4)
        end_step: Last step to run (1-4)
        resume_from: Optional checkpoint path
        
    Returns:
        Path to final checkpoint
    """
    # Get vocab_size
    vocab_size = get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Create adapters
    model_factory = make_model_factory(vocab_size)
    data_loader = make_data_loader()
    
    # Run with PTBRunner (extends Runner with ResetStatesCallback)
    runner = PTBRunner(experiment, model_factory, data_loader)
    
    return runner.run(
        start_step=start_step,
        end_step=end_step,
        resume_from=resume_from,
    )
