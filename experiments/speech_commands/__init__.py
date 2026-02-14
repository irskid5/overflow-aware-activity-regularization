"""Speech Commands experiment package.

Provides Google Speech Commands dataset support for OAR quantized RNN experiments.
"""
from experiments.speech_commands.data import (
    get_datasets,
    NUM_CLASSES,
    NUM_MFCC,
    NUM_FRAMES,
)
from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
from experiments.speech_commands.model import get_model
from experiments.speech_commands.training import run_speech_commands, make_data_loader

__all__ = [
    # Data
    "get_datasets",
    "NUM_CLASSES",
    "NUM_MFCC",
    "NUM_FRAMES",
    # Config
    "SPEECH_COMMANDS_EXPERIMENT",
    # Model
    "get_model",
    # Training
    "run_speech_commands",
    "make_data_loader",
]
