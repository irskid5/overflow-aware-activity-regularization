"""MNIST experiment configuration and training orchestration."""

import copy
import json
import os
from datetime import datetime

import tensorflow as tf
from oar import mod_sign, sign_ste_tanh

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oar.config import TrainingStepConfig


# ============================================================================
# NEW API - perform_training_steps() based orchestration
# ============================================================================


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


def perform_training_steps(
    steps: list["TrainingStepConfig"],
    step_4_factory=None,
) -> str:
    """Execute training steps."""
    from experiments.mnist.training import train, create_run_dir, tee_output
    from experiments.mnist.steps import create_step_4
    
    if not steps:
        raise ValueError("steps list cannot be empty")
    
    if step_4_factory is None:
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


# ============================================================================
# OLD API - Deprecated, will be removed in Task 8
# ============================================================================

# Default options for MNIST RNN experiment
MNIST_OPTIONS = {
    "enlarge": False,
    "epochs": 100,
    "learning_rate": 1e-4,
    "batch_size": 512,
    "t": 1.5,  # Ternarization threshold multiplier
    "tᵢ": 0.7,  # Input ternarization threshold
    "s": 4.0,  # Gradient scaling factor
    "oar": {
        "oar_lambda": 1e-4,  # OAR regularization rate
        "omega": 6,  # Bit precision (2^omega modulus)
    },
    "quantize": False,
}

# Enlarged MNIST options (128x128)
ENLARGED_MNIST_OPTIONS = {
    "enlarge": True,
    "epochs": 100,
    "learning_rate": 5e-6,
    "batch_size": 512,
    "t": 1.5,
    "tᵢ": 0.7,
    "s": 4.0,
    "oar": {
        "oar_lambda": 1e-4,
        "omega": 6,
    },
    "quantize": False,
}


def _make_oar_config(use: bool, oar_lambda: float, omega: int) -> dict:
    """Create OAR configuration dict.

    Args:
        use: Whether to enable OAR regularization
        oar_lambda: OAR regularization rate
        omega: Bit precision (2^omega modulus)

    Returns:
        Dict with OAR configuration
    """
    return {
        "use": use,
        "oar_lambda": oar_lambda,
        "omega": omega,
    }


def _save_experiment_config(run_dir: str, options: dict) -> None:
    """Save initial experiment configuration.

    Args:
        run_dir: Run directory path
        options: Initial experiment options (before any mutations)
    """
    config = {
        "started_at": datetime.now().isoformat(),
        "initial_options": copy.deepcopy(options),
        "experiment_type": "four_step_quantization",
    }

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def get_default_layer_options(options: dict) -> dict:
    """Build default layer_options dict from high-level options.

    Creates per-layer configuration for MNIST RNN architecture.
    Used as starting point for four-step quantization.

    Args:
        options: Dict with "oar" sub-dict containing "omega" and "oar_lambda"

    Returns:
        Dict mapping layer names to their configuration
    """
    oar_lambda = options["oar"]["oar_lambda"]
    omega = options["oar"]["omega"]

    result = {"INPUT": {"ternarize": False}}

    for name in ["QRNN_0", "QRNN_1", "DENSE_0"]:
        result[name] = {
            "activation": tf.keras.activations.tanh,
            "oar": _make_oar_config(use=False, oar_lambda=oar_lambda, omega=omega),
            "s": 1.0,
            "τ": 0.0,
        }

    result["DENSE_OUT"] = {
        "activation": tf.keras.activations.softmax,
        "oar": _make_oar_config(use=True, oar_lambda=0.0, omega=omega),
        "s": 1.0,
        "τ": 0.0,
    }

    return result


def get_model_parameter_stats(pretrained_weights: str | None, options, layer_options):
    """Gets the mean of the absolute value for ternarization of each parameter.

    Used to compute per-layer ternarization thresholds: τ = t * E(|θ_l|)

    Args:
        pretrained_weights: Path to checkpoints folder containing parameters
        options: Model options
        layer_options: Layer-wise model options

    Returns:
        Dict mapping layer names to mean absolute weight values
    """
    from experiments.mnist.model import get_model

    tern_params = {"QRNN_0": 0, "QRNN_1": 0, "DENSE_0": 0, "DENSE_OUT": 0}
    model = get_model(options, layer_options)

    if pretrained_weights is None:
        return tern_params

    model.load_weights(pretrained_weights)
    print("Restored pretrained weights from {}.".format(pretrained_weights))

    # Calculate weight stats per layer
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:
            all_weights = tf.concat(
                [tf.reshape(x, shape=[-1]) for x in layer.trainable_weights], axis=-1
            )
            mean_abs = tf.math.reduce_mean(tf.abs(all_weights))
            if layer.name.find("QRNN_0") != -1:
                tern_params["QRNN_0"] = mean_abs.numpy()
            if layer.name.find("QRNN_1") != -1:
                tern_params["QRNN_1"] = mean_abs.numpy()
            if layer.name.find("DENSE_0") != -1:
                tern_params["DENSE_0"] = mean_abs.numpy()
            if layer.name.find("DENSE_OUT") != -1:
                tern_params["DENSE_OUT"] = mean_abs.numpy()
    return tern_params


def perform_step_in_four_step_quant(
    step: int,
    pretrained_weights: str | None,
    options: dict,
    run_dir: str | None = None,
) -> str:
    """Performs a step in the four-step quantization procedure.

    Steps:
    1. Train with tanh activation (baseline)
    2. Replace tanh with sign (tanh gradient), apply gradient scaling s
    3. Ternarize inputs
    4. Enable weight quantization + OAR (learning rate ×0.1)

    Args:
        step: Step number (1-4)
        pretrained_weights: Path to checkpoints folder for initialization
        options: Overall options dict
        run_dir: Optional run directory. If not provided, train() creates one.

    Returns:
        Path to checkpoints folder of the trained model
    """
    # Import here to avoid circular imports
    from experiments.mnist.training import train

    print(f"\nPERFORMING STEP {step}/4 FROM FOUR-STEP QUANTIZATION PROCESS\n")

    layer_options = get_default_layer_options(options)

    # Change settings according to step number
    ternarize_inputs = False
    t = 1.0
    s = 1.0
    activation = tf.keras.activations.tanh
    oar = False
    tern_params = {"QRNN_0": 0, "QRNN_1": 0, "DENSE_0": 0, "DENSE_OUT": 0}

    if step == 2:
        s = options["s"]
        activation = sign_ste_tanh
    if step == 3:
        ternarize_inputs = True
    if step == 4:
        options["quantize"] = True
        options["learning_rate"] *= 0.1
        t = options["t"]
        oar = True
        tern_params = get_model_parameter_stats(
            pretrained_weights, options, layer_options
        )
        print("\nTERNARIZATION PARAMETERS:")
        print(tern_params)

        def activation(x):
            return mod_sign(x, num_bits=options["oar"]["omega"])

    # Adjust layer options
    oar_lambda = options["oar"]["oar_lambda"]
    omega = options["oar"]["omega"]

    layer_options = {"INPUT": {"ternarize": ternarize_inputs}}

    for name in ["QRNN_0", "QRNN_1"]:
        layer_options[name] = {
            "activation": activation,
            "oar": _make_oar_config(use=oar, oar_lambda=oar_lambda, omega=omega),
            "s": s,
            "τ": t * tern_params[name],
        }

    layer_options["DENSE_0"] = {
        "activation": activation,
        "oar": _make_oar_config(use=oar, oar_lambda=oar_lambda, omega=omega),
        "s": 1.0,
        "τ": t * tern_params["DENSE_0"],
    }

    layer_options["DENSE_OUT"] = {
        "activation": tf.keras.activations.softmax,
        "oar": _make_oar_config(use=True, oar_lambda=0.0, omega=omega),
        "s": 1.0,
        "τ": t * tern_params["DENSE_OUT"],
    }

    print("OPTIONS AND LAYER OPTIONS:")
    print(options)
    print(layer_options)

    return train(
        pretrained_weights=pretrained_weights,
        options=options,
        layer_options=layer_options,
        step=step,
        run_dir=run_dir,
    )


def perform_four_step_quant(options: dict) -> str:
    """Performs the complete four-step quantization procedure.

    Creates a single run directory and executes all four steps within it.

    Args:
        options: Experiment options dict

    Returns:
        Path to checkpoint folder of final step parameters
    """
    from experiments.mnist.training import create_run_dir, tee_output

    # Create shared run directory for all steps
    run_dir = create_run_dir()

    # Save initial experiment config (before any mutations)
    _save_experiment_config(run_dir, options)

    # Log all output to experiment-level log file
    log_path = os.path.join(run_dir, "output.log")

    with tee_output(log_path):
        pretrained_weights = None
        for step in range(1, 5):
            pretrained_weights = perform_step_in_four_step_quant(
                step=step,
                pretrained_weights=pretrained_weights,
                options=options,
                run_dir=run_dir,
            )
            options["epochs"] = 1000

    return pretrained_weights
