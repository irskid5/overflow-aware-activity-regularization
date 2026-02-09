"""MNIST experiment configuration and training orchestration."""

import json
import os
from datetime import datetime

import tensorflow as tf

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oar.config import TrainingStepConfig


def get_model_parameter_stats_new(
    pretrained_weights: str | None,
    step_config: "TrainingStepConfig",
    layer_names: list[str] | None = None,
) -> dict[str, float]:
    """Get mean absolute weights per layer for threshold computation."""
    from experiments.mnist.model import get_model
    
    if layer_names is None:
        layer_names = ["QRNN_0", "QRNN_1", "DENSE_0", "DENSE_OUT"]
    
    tern_params = {name: 0.0 for name in layer_names}
    
    if pretrained_weights is None:
        return tern_params
    
    model = get_model(step_config)
    model.load_weights(pretrained_weights)
    print(f"Loaded weights from {pretrained_weights} for threshold computation.")
    
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:
            all_weights = tf.concat(
                [tf.reshape(w, [-1]) for w in layer.trainable_weights], axis=-1
            )
            mean_abs = float(tf.reduce_mean(tf.abs(all_weights)).numpy())
            
            for name in layer_names:
                if name in layer.name:
                    tern_params[name] = mean_abs
                    break
    
    return tern_params


def _save_experiment_config_new(run_dir: str, steps: list["TrainingStepConfig"]) -> None:
    """Save experiment configuration with TrainingStepConfig list."""
    config = {
        "started_at": datetime.now().isoformat(),
        "experiment_type": "four_step_quantization",
        "steps": [asdict(s) for s in steps],
    }
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


_USE_DEFAULT = object()


def perform_training_steps(
    steps: list["TrainingStepConfig"],
    step_4_factory=_USE_DEFAULT,
) -> str:
    """Execute training steps.
    
    Args:
        steps: List of TrainingStepConfig to execute sequentially
        step_4_factory: Factory function for step 4. Defaults to create_step_4.
                       Pass None to skip step 4 entirely.
    """
    from experiments.mnist.training import train, create_run_dir, tee_output
    from experiments.mnist.steps import create_step_4
    
    if not steps:
        raise ValueError("steps list cannot be empty")
    
    if step_4_factory is _USE_DEFAULT:
        step_4_factory = create_step_4
    
    run_dir = create_run_dir()
    log_path = os.path.join(run_dir, "output.log")
    
    with tee_output(log_path):
        pretrained_weights = None
        all_steps = list(steps)
        total = len(steps) + (1 if step_4_factory else 0)
        
        for i, step_config in enumerate(steps, start=1):
            print(f"\nPERFORMING STEP {i}/{total}: {step_config.name}\n")
            
            pretrained_weights = train(
                step_config=step_config,
                step_number=i,
                pretrained_weights=pretrained_weights,
                run_dir=run_dir,
            )
        
        if step_4_factory:
            tern_params = get_model_parameter_stats_new(pretrained_weights, steps[-1])
            print("\nTERNARIZATION PARAMETERS:")
            print(tern_params)
            
            step_4 = step_4_factory(tern_params)
            all_steps.append(step_4)
            print(f"\nPERFORMING STEP {len(steps)+1}/{total}: {step_4.name}\n")
            
            pretrained_weights = train(
                step_config=step_4,
                step_number=len(steps) + 1,
                pretrained_weights=pretrained_weights,
                run_dir=run_dir,
            )
        
        _save_experiment_config_new(run_dir, all_steps)
    
    return pretrained_weights
