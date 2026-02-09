"""Entry point for OAR experiments."""

import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_jit("autoclustering")

tf.random.set_seed(1997)  # For experimental reproducibility

from experiments.mnist import perform_training_steps, STATIC_STEPS, create_step_4
from experiments.mnist.steps import STEP_1, STEP_2, STEP_3, compute_thresholds
from oar.config import TrainingStepConfig, LayerConfig


def main():
    """Runs four-step quantization using the new config-driven API."""
    final_checkpoint = perform_training_steps(
        steps=STATIC_STEPS,
        step_4_factory=create_step_4,
    )
    print(f"\nTraining complete. Final checkpoint: {final_checkpoint}")
    return final_checkpoint


def evaluation_with_and_without_oar2():
    """Evaluates models with and without OAR2 across different bit widths.
    
    Trains steps 1-3 once, then runs step 4 with varying omega (3-8) and
    oar_lambda (0 vs 1e-3) to compare performance with and without OAR.
    """
    from experiments.mnist.config import get_model_parameter_stats_new
    from experiments.mnist.training import train, create_run_dir, tee_output
    
    run_dir = create_run_dir()
    log_path = os.path.join(run_dir, "output.log")
    
    with tee_output(log_path):
        # Train steps 1-3
        pretrained_weights = None
        steps_1_to_3 = [STEP_1, STEP_2, STEP_3]
        
        for i, step_config in enumerate(steps_1_to_3, start=1):
            print(f"\nPERFORMING STEP {i}/3: {step_config.name}\n")
            pretrained_weights = train(
                step_config=step_config,
                step_number=i,
                pretrained_weights=pretrained_weights,
                run_dir=run_dir,
            )
        
        # Get ternarization parameters from step 3
        tern_params = get_model_parameter_stats_new(pretrained_weights, STEP_3)
        print("\nTERNARIZATION PARAMETERS:")
        print(tern_params)
        
        # Run step 4 with varying omega and oar_lambda
        for oar_lambda in [0.0, 1e-3]:
            for omega in range(3, 9):
                print(f"\n=== EVALUATION: omega={omega}, oar_lambda={oar_lambda} ===\n")
                
                thresholds = compute_thresholds(tern_params, t=1.5)
                
                step_4 = TrainingStepConfig(
                    name=f"step_4_omega{omega}_lambda{oar_lambda}",
                    epochs=1000,
                    learning_rate=1e-5,
                    batch_size=512,
                    layers={
                        "QRNN_0": LayerConfig(
                            activation="mod_sign",
                            gradient_scale=4.0,
                            oar_lambda=oar_lambda if oar_lambda > 0 else None,
                            omega=omega,
                            quantize_threshold=thresholds.get("QRNN_0"),
                        ),
                        "QRNN_1": LayerConfig(
                            activation="mod_sign",
                            gradient_scale=4.0,
                            oar_lambda=oar_lambda if oar_lambda > 0 else None,
                            omega=omega,
                            quantize_threshold=thresholds.get("QRNN_1"),
                        ),
                        "DENSE_0": LayerConfig(
                            activation="mod_sign",
                            oar_lambda=oar_lambda if oar_lambda > 0 else None,
                            omega=omega,
                            quantize_threshold=thresholds.get("DENSE_0"),
                        ),
                        "DENSE_OUT": LayerConfig(
                            activation="softmax",
                            omega=omega,
                            quantize_threshold=thresholds.get("DENSE_OUT"),
                        ),
                    },
                )
                
                train(
                    step_config=step_4,
                    step_number=4,
                    pretrained_weights=pretrained_weights,
                    run_dir=run_dir,
                )


def evaluation_different_oar_regularization_rates():
    """Evaluates different OAR regularization rates.
    
    Trains steps 1-3 once, then runs step 4 with varying oar_lambda values
    at omega=5 and omega=6 to find optimal regularization strength.
    """
    from experiments.mnist.config import get_model_parameter_stats_new
    from experiments.mnist.training import train, create_run_dir, tee_output
    
    run_dir = create_run_dir()
    log_path = os.path.join(run_dir, "output.log")
    
    with tee_output(log_path):
        # Train steps 1-3
        pretrained_weights = None
        steps_1_to_3 = [STEP_1, STEP_2, STEP_3]
        
        for i, step_config in enumerate(steps_1_to_3, start=1):
            print(f"\nPERFORMING STEP {i}/3: {step_config.name}\n")
            pretrained_weights = train(
                step_config=step_config,
                step_number=i,
                pretrained_weights=pretrained_weights,
                run_dir=run_dir,
            )
        
        # Get ternarization parameters from step 3
        tern_params = get_model_parameter_stats_new(pretrained_weights, STEP_3)
        print("\nTERNARIZATION PARAMETERS:")
        print(tern_params)
        
        # Test different regularization rates
        rates = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        bits = [5, 6]
        
        for omega in bits:
            for oar_lambda in rates:
                print(f"\n=== EVALUATION: omega={omega}, oar_lambda={oar_lambda} ===\n")
                
                thresholds = compute_thresholds(tern_params, t=1.5)
                
                step_4 = TrainingStepConfig(
                    name=f"step_4_omega{omega}_lambda{oar_lambda}",
                    epochs=1000,
                    learning_rate=1e-5,
                    batch_size=512,
                    layers={
                        "QRNN_0": LayerConfig(
                            activation="mod_sign",
                            gradient_scale=4.0,
                            oar_lambda=oar_lambda if oar_lambda > 0 else None,
                            omega=omega,
                            quantize_threshold=thresholds.get("QRNN_0"),
                        ),
                        "QRNN_1": LayerConfig(
                            activation="mod_sign",
                            gradient_scale=4.0,
                            oar_lambda=oar_lambda if oar_lambda > 0 else None,
                            omega=omega,
                            quantize_threshold=thresholds.get("QRNN_1"),
                        ),
                        "DENSE_0": LayerConfig(
                            activation="mod_sign",
                            oar_lambda=oar_lambda if oar_lambda > 0 else None,
                            omega=omega,
                            quantize_threshold=thresholds.get("DENSE_0"),
                        ),
                        "DENSE_OUT": LayerConfig(
                            activation="softmax",
                            omega=omega,
                            quantize_threshold=thresholds.get("DENSE_OUT"),
                        ),
                    },
                )
                
                train(
                    step_config=step_4,
                    step_number=4,
                    pretrained_weights=pretrained_weights,
                    run_dir=run_dir,
                )


if __name__ == "__main__":
    main()
    # evaluation_with_and_without_oar2()
    # evaluation_different_oar_regularization_rates()
