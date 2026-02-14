"""PTB experiment configuration for four-step quantization with OAR.

Architecture: Embedding → QRNN_0 → QRNN_1 → DENSE_OUT
- No Downsampling (unlike MNIST)
- No DENSE_0 hidden layer
- Stateful RNN for language modeling
"""
from dacite import from_dict

from oar.config import DACITE_CONFIG, ExperimentConfig

# PTB-specific constants
BATCH_SIZE = 20  # Standard for PTB (Zaremba et al.)
NUM_STEPS = 25  # Sequence length (matching SHE paper)
EMBED_DIM = 650  # Hidden dimension (SHE uses 1300 with 1 layer; we use 2 layers)

PTB_CONFIG = {
    "name": "ptb_four_step",
    "layer_names": ["INPUT", "QRNN_0", "QRNN_1", "DENSE_OUT"],
    "runs_dir": "runs/ptb/",
    "steps": {
        # Step 1: Tanh baseline
        1: {
            "name": "tanh_baseline",
            "epochs": 50,
            "learning_rate": 1e-3,
            "batch_size": BATCH_SIZE,
            "layers": {
                "INPUT": {},
                "QRNN_0": {"activation": "tanh"},
                "QRNN_1": {"activation": "tanh"},
                "DENSE_OUT": {"activation": "softmax"},
            },
        },
        # Step 2: Sign activation with gradient scaling
        2: {
            "name": "sign_activation",
            "epochs": 100,
            "learning_rate": 1e-4,
            "batch_size": BATCH_SIZE,
            "layers": {
                "INPUT": {},
                "QRNN_0": {
                    "activation": {"function": "sign_ste_tanh", "gradient_scale": 4.0}
                },
                "QRNN_1": {
                    "activation": {"function": "sign_ste_tanh", "gradient_scale": 4.0}
                },
                "DENSE_OUT": {"activation": "softmax"},
            },
        },
        # Step 3: Input quantization (ternarize embedding output)
        3: {
            "name": "input_quantization",
            "epochs": 50,
            "learning_rate": 1e-4,
            "batch_size": BATCH_SIZE,
            "layers": {
                "INPUT": {"quantization": {"ternarization_scale": 0.7}},
                "QRNN_0": {
                    "activation": {"function": "sign_ste_tanh", "gradient_scale": 4.0}
                },
                "QRNN_1": {
                    "activation": {"function": "sign_ste_tanh", "gradient_scale": 4.0}
                },
                "DENSE_OUT": {"activation": "softmax"},
            },
        },
        # Step 4: Full quantization + OAR
        4: {
            "name": "full_quantization",
            "epochs": 200,
            "learning_rate": 1e-5,
            "cosine_decay_epochs": 200,
            "batch_size": BATCH_SIZE,
            "layers": {
                "INPUT": {"quantization": {"ternarization_scale": 0.7}},
                "QRNN_0": {
                    "activation": {
                        "function": "mod_sign",
                        "gradient_scale": 4.0,
                        "omega": 6,
                    },
                    "quantization": {
                        "ternarization_scale": 1.5,
                        "oar": {"regularization_rate": 1e-4, "omega": 6},
                    },
                },
                "QRNN_1": {
                    "activation": {
                        "function": "mod_sign",
                        "gradient_scale": 4.0,
                        "omega": 6,
                    },
                    "quantization": {
                        "ternarization_scale": 1.5,
                        "oar": {"regularization_rate": 1e-4, "omega": 6},
                    },
                },
                "DENSE_OUT": {
                    "activation": "softmax",
                    "quantization": {"ternarization_scale": 1.5},
                },
            },
        },
    },
}

PTB_EXPERIMENT = from_dict(ExperimentConfig, PTB_CONFIG, config=DACITE_CONFIG)
