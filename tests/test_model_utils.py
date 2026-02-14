"""Tests for oar.model_utils."""
import tensorflow as tf


def test_compute_thresholds_if_needed_returns_empty_when_no_thresholds_needed():
    """Returns empty dict when no layers need threshold computation."""
    from oar.model_utils import compute_thresholds_if_needed
    from oar.config import StepConfig, LayerStepConfig
    
    # Step config with no ternarization_scale
    step_config = StepConfig(
        name="test",
        layers={
            "LAYER_0": LayerStepConfig(),  # defaults, no quantization
        }
    )
    
    result = compute_thresholds_if_needed(step_config, None, lambda sc, th: None)
    assert result == {}


def test_compute_thresholds_if_needed_skips_input_layer():
    """INPUT layer is skipped even with ternarization_scale set."""
    from oar.model_utils import compute_thresholds_if_needed
    from oar.config import StepConfig, LayerStepConfig, QuantizationConfig
    
    step_config = StepConfig(
        name="test",
        layers={
            "INPUT": LayerStepConfig(
                quantization=QuantizationConfig(ternarization_scale=0.7)
            ),
        }
    )
    
    result = compute_thresholds_if_needed(step_config, None, lambda sc, th: None)
    assert result == {}
