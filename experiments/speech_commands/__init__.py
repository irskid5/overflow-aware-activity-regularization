"""Speech Commands experiment package."""
from experiments.speech_commands.data import get_datasets, NUM_CLASSES, NUM_MFCC, NUM_FRAMES
from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT

__all__ = [
    "get_datasets",
    "NUM_CLASSES",
    "NUM_MFCC", 
    "NUM_FRAMES",
    "SPEECH_COMMANDS_EXPERIMENT",
]
