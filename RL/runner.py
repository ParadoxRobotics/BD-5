"""Runs training and evaluation loop for the Z-Bot."""

import argparse

from common import randomize
from common.runner import BaseRunner
import joystick


class BD5Runner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        self.env_config = joystick.default_config()
        self.env = joystick.Joystick(task=args.task)
        self.eval_env = joystick.Joystick(task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        print(f"Observation size: {self.obs_size}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    runner = BD5Runner(args)

    runner.train()

if __name__ == "__main__":
    main()