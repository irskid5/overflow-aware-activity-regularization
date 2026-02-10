"""Runner for multi-step quantization experiments.

Provides the Runner class that orchestrates training steps, managing:
- Step sequencing with configurable start/end steps
- Checkpoint management between steps
- Output logging (console + file via tee_output)
- Config persistence (experiment + per-step)

The Runner delegates model construction (including threshold computation)
to a user-provided model_factory, keeping training orchestration separate
from model architecture details.

Example usage:
    runner = Runner(experiment_config, model_factory, data_loader)
    final_checkpoint = runner.run(start_step=1, end_step=4)
"""

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Protocol

import tensorflow as tf

from oar.config import ExperimentConfig, StepConfig


class ModelFactory(Protocol):
    """Protocol for model factory functions.
    
    The model factory is responsible for:
    - Building the model architecture for a given step config
    - Computing thresholds from pretrained_weights if needed
    - Applying weight quantization based on step config
    """

    def __call__(
        self,
        step_config: StepConfig,
        pretrained_weights: str | None = None,
    ) -> tf.keras.Model:
        """Build model, computing thresholds from pretrained_weights if needed.
        
        Args:
            step_config: Configuration for the training step
            pretrained_weights: Path to checkpoint for loading weights and
                               computing thresholds (e.g., τ = t × E[|θ|])
                               
        Returns:
            Compiled or uncompiled Keras model ready for training
        """
        ...


class DataLoader(Protocol):
    """Protocol for data loader functions."""

    def __call__(
        self,
        batch_size: int,
        enlarge: bool,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Return (train, val, test) datasets.
        
        Args:
            batch_size: Batch size for all datasets
            enlarge: Whether to use enlarged inputs (e.g., 128x128 vs 28x28 for MNIST)
            
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        ...


class Runner:
    """Runs multi-step quantization experiments.
    
    Responsibilities:
    - Orchestrate training steps
    - Save configs and checkpoints
    - Log output to file
    
    NOT responsible for:
    - Threshold computation (delegated to model_factory)
    - TensorFlow internals
    
    Example:
        >>> experiment = ExperimentConfig(
        ...     name="mnist_4step",
        ...     layer_names=["INPUT", "QRNN_0", "QRNN_1", "DENSE_0", "DENSE_OUT"],
        ...     steps={1: step1_config, 2: step2_config, ...}
        ... )
        >>> runner = Runner(experiment, model_factory, data_loader)
        >>> final_ckpt = runner.run(start_step=1, end_step=4)
    """

    def __init__(
        self,
        experiment: ExperimentConfig,
        model_factory: ModelFactory,
        data_loader: DataLoader,
    ):
        """Initialize the runner.
        
        Args:
            experiment: Full experiment configuration with steps
            model_factory: Callable that builds models for each step
            data_loader: Callable that returns train/val/test datasets
        """
        self.experiment = experiment
        self.model_factory = model_factory
        self.data_loader = data_loader

    def run(
        self,
        start_step: int = 1,
        end_step: int | None = None,
        resume_from: str | None = None,
    ) -> str:
        """Run steps from start_step to end_step.
        
        Args:
            start_step: First step to run (1-indexed)
            end_step: Last step to run (inclusive, default: max step)
            resume_from: Checkpoint path to load weights from
            
        Returns:
            Path to final checkpoint
            
        Raises:
            ValueError: If start_step > end_step or step not defined
        """
        if end_step is None:
            end_step = max(self.experiment.steps.keys())

        if start_step > end_step:
            raise ValueError(f"start_step ({start_step}) > end_step ({end_step})")

        for step_num in range(start_step, end_step + 1):
            if step_num not in self.experiment.steps:
                raise ValueError(f"Step {step_num} not defined in experiment")

        run_dir = self._create_run_dir()
        log_path = os.path.join(run_dir, "output.log")
        self._save_experiment_config(run_dir)

        pretrained_weights = resume_from
        total_steps = end_step - start_step + 1

        with tee_output(log_path):
            for i, step_num in enumerate(range(start_step, end_step + 1), start=1):
                step_config = self.experiment.steps[step_num]

                print(f"\n{'='*60}")
                print(f"STEP {step_num} ({i}/{total_steps}): {step_config.name}")
                print(f"  epochs={step_config.epochs}, lr={step_config.learning_rate}")
                print(f"{'='*60}\n")

                pretrained_weights = self._run_step(
                    step_num=step_num,
                    step_config=step_config,
                    pretrained_weights=pretrained_weights,
                    run_dir=run_dir,
                )

        return pretrained_weights

    def _run_step(
        self,
        step_num: int,
        step_config: StepConfig,
        pretrained_weights: str | None,
        run_dir: str,
    ) -> str:
        """Execute a single training step.
        
        Args:
            step_num: Step number (for directory naming)
            step_config: Configuration for this step
            pretrained_weights: Path to load weights from (or None for step 1)
            run_dir: Base run directory
            
        Returns:
            Path to checkpoint saved after this step
        """
        from oar import ReservoirHistogramCallback, reset_stat_weights

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

        # Compile with LR scheduler
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=step_config.learning_rate,
            decay_steps=step_config.epochs * len(ds_train),
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        callbacks = [
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
        model.evaluate(ds_test, verbose=2)

        checkpoint_path = os.path.join(output_dir, "checkpoints/")
        self._save_step_config(output_dir, step_num, step_config, checkpoint_path)

        return checkpoint_path

    def _create_run_dir(self) -> str:
        """Create timestamped run directory.
        
        Returns:
            Path to created run directory
        """
        now = datetime.now()
        run_dir = os.path.join(
            self.experiment.runs_dir,
            now.strftime("%Y%m%d-%H%M%S"),
        )
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _save_experiment_config(self, run_dir: str) -> None:
        """Save experiment config as JSON.
        
        Args:
            run_dir: Directory to save config in
        """
        config_path = os.path.join(run_dir, "experiment_config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.experiment), f, indent=2)

    def _save_step_config(
        self,
        output_dir: str,
        step_num: int,
        step_config: StepConfig,
        checkpoint_path: str,
    ) -> None:
        """Save step config and checkpoint path as JSON.
        
        Args:
            output_dir: Step output directory
            step_num: Step number
            step_config: Step configuration
            checkpoint_path: Path to saved checkpoint
        """
        config = {
            "step": step_num,
            "config": asdict(step_config),
            "checkpoint": checkpoint_path,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)


@contextmanager
def tee_output(log_path: str):
    """Tee stdout/stderr to both console and log file.
    
    Args:
        log_path: Path to log file
        
    Yields:
        None (context manager)
        
    Example:
        >>> with tee_output("output.log"):
        ...     print("This goes to both console and file")
    """

    class TeeWriter:
        """Writer that duplicates output to original stream and log file."""

        def __init__(self, original, log_file):
            self.original = original
            self.log_file = log_file
            self.encoding = getattr(original, "encoding", "utf-8")

        def write(self, data):
            self.original.write(data)
            self.log_file.write(data)
            self.log_file.flush()

        def flush(self):
            self.original.flush()
            self.log_file.flush()

        def isatty(self):
            return False

        def fileno(self):
            return self.original.fileno()

    with open(log_path, "w") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeWriter(old_stdout, log_file)
        sys.stderr = TeeWriter(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
