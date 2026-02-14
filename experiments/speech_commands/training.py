"""Speech Commands training utilities.

Adapts the oar.runner.Runner for Speech Commands experiment.
"""
from typing import Callable


from experiments.speech_commands.config import SPEECH_COMMANDS_EXPERIMENT
from experiments.speech_commands.data import get_datasets
from experiments.speech_commands.model import get_model
from oar.config import ExperimentConfig
from oar.runner import Runner


def make_data_loader() -> Callable:
    """Create DataLoader-compatible callable.
    
    Returns:
        Callable matching (batch_size, enlarge) -> (train, val, test)
    """
    # get_datasets already matches DataLoader protocol signature
    return get_datasets


def run_speech_commands(
    experiment: ExperimentConfig | None = None,
    start_step: int = 1,
    end_step: int = 4,
    resume_from: str | None = None,
) -> str:
    """Run Speech Commands experiment.
    
    Args:
        experiment: Experiment config (defaults to SPEECH_COMMANDS_EXPERIMENT)
        start_step: First step to run (1-4)
        end_step: Last step to run (1-4)
        resume_from: Optional checkpoint path
        
    Returns:
        Path to final checkpoint
    """
    if experiment is None:
        experiment = SPEECH_COMMANDS_EXPERIMENT
    
    runner = Runner(experiment, get_model, make_data_loader())
    
    return runner.run(
        start_step=start_step,
        end_step=end_step,
        resume_from=resume_from,
    )
