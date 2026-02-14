#!/usr/bin/env python
"""Run Speech Commands four-step quantization experiment.

Usage:
    python run_speech_commands.py                    # Run all steps
    python run_speech_commands.py --start 2 --end 4  # Run steps 2-4
    python run_speech_commands.py --resume path/to/checkpoint  # Resume
"""
import argparse

import tensorflow as tf

from experiments.speech_commands import run_speech_commands, SPEECH_COMMANDS_EXPERIMENT


def main():
    parser = argparse.ArgumentParser(description="Speech Commands OAR experiment")
    parser.add_argument("--start", type=int, default=1, help="Start step (1-4)")
    parser.add_argument("--end", type=int, default=4, help="End step (1-4)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Set seed for reproducibility
    tf.random.set_seed(1997)

    print(f"Running Speech Commands experiment: steps {args.start}-{args.end}")
    print(f"Config: {SPEECH_COMMANDS_EXPERIMENT.name}")

    final_checkpoint = run_speech_commands(
        start_step=args.start,
        end_step=args.end,
        resume_from=args.resume,
    )

    print(f"\nExperiment complete. Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
