"""MNIST four-step quantization configuration."""

from oar.config import LayerConfig, InputConfig, TrainingStepConfig


def get_default_layer_config(layer_name: str) -> LayerConfig:
    """Get default LayerConfig for a layer.
    
    DENSE_OUT uses softmax, all others use tanh.
    """
    if layer_name == "DENSE_OUT":
        return LayerConfig(activation="softmax")
    return LayerConfig()


def compute_thresholds(
    tern_params: dict[str, float],
    t: float = 1.5,
) -> dict[str, float]:
    """Compute ternarization thresholds: τ = t × E[|θ|].
    
    Args:
        tern_params: Mean absolute weights per layer
        t: Threshold multiplier
        
    Returns:
        Thresholds per layer
    """
    return {name: t * mean_abs for name, mean_abs in tern_params.items()}


# Step 1: Tanh baseline - all defaults
STEP_1 = TrainingStepConfig(
    name="tanh_baseline",
    epochs=100,
)

# Step 2: Sign activation with gradient scaling for RNNs
STEP_2 = TrainingStepConfig(
    name="sign_with_gradient_scaling",
    epochs=1000,
    layers={
        "QRNN_0": LayerConfig(activation="sign_ste_tanh", gradient_scale=4.0),
        "QRNN_1": LayerConfig(activation="sign_ste_tanh", gradient_scale=4.0),
        "DENSE_0": LayerConfig(activation="sign_ste_tanh"),
    },
)

# Step 3: Quantize inputs
STEP_3 = TrainingStepConfig(
    name="quantize_inputs",
    epochs=1000,
    input_config=InputConfig(quantize_threshold=0.7),
    layers={
        "QRNN_0": LayerConfig(activation="sign_ste_tanh", gradient_scale=4.0),
        "QRNN_1": LayerConfig(activation="sign_ste_tanh", gradient_scale=4.0),
        "DENSE_0": LayerConfig(activation="sign_ste_tanh"),
    },
)


def create_step_4(tern_params: dict[str, float], t: float = 1.5) -> TrainingStepConfig:
    """Create Step 4 config with computed thresholds.
    
    Step 4 needs runtime threshold computation: τ = t × E[|θ|]
    
    Args:
        tern_params: Mean absolute weights per layer from previous step
        t: Threshold multiplier (default 1.5)
        
    Returns:
        TrainingStepConfig for step 4
    """
    thresholds = compute_thresholds(tern_params, t)
    
    return TrainingStepConfig(
        name="full_quantization_with_oar",
        epochs=1000,
        learning_rate=1e-5,
        input_config=InputConfig(quantize_threshold=0.7),
        layers={
            "QRNN_0": LayerConfig(
                activation="mod_sign",
                gradient_scale=4.0,
                oar_lambda=1e-4,
                omega=6,
                quantize_threshold=thresholds.get("QRNN_0", 0),
            ),
            "QRNN_1": LayerConfig(
                activation="mod_sign",
                gradient_scale=4.0,
                oar_lambda=1e-4,
                omega=6,
                quantize_threshold=thresholds.get("QRNN_1", 0),
            ),
            "DENSE_0": LayerConfig(
                activation="mod_sign",
                oar_lambda=1e-4,
                omega=6,
                quantize_threshold=thresholds.get("DENSE_0", 0),
            ),
            "DENSE_OUT": LayerConfig(
                activation="softmax",
                oar_lambda=0.0,  # Track stats only
                omega=6,
                quantize_threshold=thresholds.get("DENSE_OUT", 0),
            ),
        },
    )


# Static steps (1-3) that don't need runtime computation
STATIC_STEPS = [STEP_1, STEP_2, STEP_3]
