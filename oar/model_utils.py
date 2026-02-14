"""Shared model utilities for OAR experiments.

Contains threshold computation and other model-building helpers.
"""
from typing import Callable

import tensorflow as tf

from oar.config import StepConfig


def compute_thresholds_if_needed(
    step_config: StepConfig,
    checkpoint_path: str | None,
    build_model_fn: Callable[[StepConfig, dict[str, float]], tf.keras.Model],
) -> dict[str, float]:
    """Compute thresholds: τ = ternarization_scale × E[|θ|].
    
    Only computes for layers that have ternarization_scale set but no explicit threshold.
    Skips INPUT layer (its threshold is used directly, not computed from weights).
    
    Args:
        step_config: Step configuration with layer configs
        checkpoint_path: Path to load pretrained weights from
        build_model_fn: Function to build model: (step_config, thresholds) -> Model
        
    Returns:
        Dictionary mapping layer names to computed thresholds
    """
    # First check if any layer needs computed thresholds
    layers_needing_thresholds = []
    for name, cfg in step_config.layers.items():
        q = cfg.quantization
        if q.ternarization_scale is not None and q.threshold is None and name != "INPUT":
            layers_needing_thresholds.append(name)
    
    if not layers_needing_thresholds or checkpoint_path is None:
        return {}
    
    # Build temp model and load weights
    temp_model = build_model_fn(step_config, {})
    temp_model.load_weights(checkpoint_path)
    
    # Compute thresholds
    thresholds = {}
    for layer_name in layers_needing_thresholds:
        t = step_config.layers[layer_name].quantization.ternarization_scale
        
        for layer in temp_model.layers:
            if layer_name in layer.name and layer.trainable_weights:
                all_weights = tf.concat(
                    [tf.reshape(w, [-1]) for w in layer.trainable_weights], axis=-1
                )
                mean_abs = float(tf.reduce_mean(tf.abs(all_weights)).numpy())
                thresholds[layer_name] = t * mean_abs
                break
    
    print(f"Computed thresholds: {thresholds}")
    return thresholds
