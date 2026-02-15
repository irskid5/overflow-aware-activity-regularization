"""Speech Commands experiment configuration.

Four-step quantization matching MNIST experiment structure.
Architecture: MFCC input → QRNN_0 → Downsampling → QRNN_1 → DENSE_0 → DENSE_OUT
"""
from dacite import from_dict
from oar.config import ExperimentConfig, DACITE_CONFIG

# Shared layer configs (same as MNIST)
_QRNN_SIGN = {"activation": {"function": "sign_ste_tanh", "gradient_scale": 4.0}}
_DENSE_SIGN = {"activation": "sign_ste_tanh"}
_DENSE_OUT = {"activation": "softmax"}
_INPUT_QUANT = {"quantization": {"ternarization_scale": 0.7}}

_QRNN_FULL = {
    "activation": {"function": "mod_sign", "gradient_scale": 4.0, "omega": 6},
    "quantization": {"ternarization_scale": 1.5, "oar": {"regularization_rate": 1e-4, "omega": 6}},
}
_DENSE_FULL = {
    "activation": {"function": "mod_sign", "omega": 6},
    "quantization": {"ternarization_scale": 1.5, "oar": {"regularization_rate": 1e-4, "omega": 6}},
}
_DENSE_OUT_FULL = {
    "activation": "softmax",
    "quantization": {"ternarization_scale": 1.5, "oar": {"regularization_rate": 0.0, "omega": 6}},
}

SPEECH_COMMANDS_CONFIG = {
    "name": "speech_commands_four_step",
    "layer_names": ["INPUT", "QRNN_0", "QRNN_1", "DENSE_0", "DENSE_OUT"],
    "runs_dir": "runs/speech_commands/",
    "steps": {
        1: {
            "name": "tanh_baseline",
            "epochs": 50,
            "batch_size": 512,
            "layers": {
                "QRNN_0": {"activation": "tanh"},
                "QRNN_1": {"activation": "tanh"},
                "DENSE_0": {"activation": "tanh"},
                "DENSE_OUT": _DENSE_OUT,
            },
        },
        2: {
            "name": "sign_activation",
            "epochs": 100,
            "batch_size": 512,
            "layers": {
                "QRNN_0": _QRNN_SIGN,
                "QRNN_1": _QRNN_SIGN,
                "DENSE_0": _DENSE_SIGN,
                "DENSE_OUT": _DENSE_OUT,
            },
        },
        3: {
            "name": "input_quantization",
            "epochs": 50,
            "batch_size": 512,
            "layers": {
                "INPUT": _INPUT_QUANT,
                "QRNN_0": _QRNN_SIGN,
                "QRNN_1": _QRNN_SIGN,
                "DENSE_0": _DENSE_SIGN,
                "DENSE_OUT": _DENSE_OUT,
            },
        },
        4: {
            "name": "full_quantization",
            "epochs": 200,
            "batch_size": 512,
            "learning_rate": 1e-4,
            "cosine_decay_epochs": 100,
            "layers": {
                "INPUT": _INPUT_QUANT,
                "QRNN_0": _QRNN_FULL,
                "QRNN_1": _QRNN_FULL,
                "DENSE_0": _DENSE_FULL,
                "DENSE_OUT": _DENSE_OUT_FULL,
            },
        },
    },
}

SPEECH_COMMANDS_EXPERIMENT = from_dict(
    ExperimentConfig, SPEECH_COMMANDS_CONFIG, config=DACITE_CONFIG
)
