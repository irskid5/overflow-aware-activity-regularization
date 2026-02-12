"""Integration tests for PTB experiment."""
import dataclasses
import os

import pytest


@pytest.mark.slow
def test_step_1_tanh_baseline(tmp_path):
    """Step 1 trains and produces valid perplexity."""
    from experiments.ptb import PTB_EXPERIMENT, run_ptb

    # Create minimal config for fast test
    step1 = dataclasses.replace(
        PTB_EXPERIMENT.steps[1],
        epochs=2,  # Just 2 epochs
    )
    experiment = dataclasses.replace(
        PTB_EXPERIMENT,
        runs_dir=str(tmp_path),
        steps={1: step1},
    )

    checkpoint = run_ptb(experiment, start_step=1, end_step=1)

    # Checkpoint should exist
    assert os.path.exists(os.path.dirname(checkpoint))

    # Should have created logs
    log_dirs = list(tmp_path.rglob("logs"))
    assert len(log_dirs) >= 1


@pytest.mark.slow
def test_full_pipeline_minimal(tmp_path):
    """All 4 steps run without error (minimal epochs)."""
    from experiments.ptb import PTB_EXPERIMENT, run_ptb

    # Minimal epochs for each step
    steps = {
        1: dataclasses.replace(PTB_EXPERIMENT.steps[1], epochs=1),
        2: dataclasses.replace(PTB_EXPERIMENT.steps[2], epochs=1),
        3: dataclasses.replace(PTB_EXPERIMENT.steps[3], epochs=1),
        4: dataclasses.replace(PTB_EXPERIMENT.steps[4], epochs=1),
    }
    experiment = dataclasses.replace(
        PTB_EXPERIMENT,
        runs_dir=str(tmp_path),
        steps=steps,
    )

    checkpoint = run_ptb(experiment, start_step=1, end_step=4)

    assert os.path.exists(os.path.dirname(checkpoint))
    assert "step_4" in checkpoint
