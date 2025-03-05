"""Defines a common runner between the different robots."""

import argparse
import functools
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from jaxtyping import PRNGKeyArray
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env


@dataclass
class RunnerConfig:
    env_config: config_dict.ConfigDict
    env: mjx_env.MjxEnv
    eval_env: mjx_env.MjxEnv
    randomizer: Callable[[mjx.Model, PRNGKeyArray], tuple[mjx.Model, jnp.ndarray]]


@dataclass
class TrainingState:
    x_data: list
    y_data: list
    y_dataerr: list
    times: list[datetime]
    rl_config: config_dict.ConfigDict
    params: Optional[Any] = None


@dataclass
class TrainingConfig:
    num_timesteps: int
    num_evals: int
    reward_scaling: float
    episode_length: int
    normalize_observations: bool
    action_repeat: int
    unroll_length: int
    num_minibatches: int
    num_updates_per_batch: int
    discounting: float
    learning_rate: float
    entropy_cost: float
    num_envs: int
    batch_size: int
    max_grad_norm: float
    clipping_epsilon: float
    num_resets_per_eval: int
    network_factory: config_dict.ConfigDict


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        """Initialize the ZBotRunner class.

        Args:
            args (argparse.Namespace): Command line arguments.
            logger (logging.Logger): Logger instance.
        """
        self.logger = logger
        self.args = args
        self.env_name = args.env
        self.base_body = "root"

        # Initialize environment
        runner_config = self.setup_environment(args.task)
        self.env_config = runner_config.env_config
        self.env = runner_config.env
        self.eval_env = runner_config.eval_env
        self.randomizer = runner_config.randomizer

        # Initialize training state
        self.training_config = self._create_training_config()
        self.training_state = TrainingState(
            x_data=[],
            y_data=[],
            y_dataerr=[],
            times=[datetime.now()],
            rl_config=config_dict.create(**self._get_rl_config_dict()),
        )

    @abstractmethod
    def setup_environment(self) -> RunnerConfig: ...

    def _create_training_config(self) -> TrainingConfig:
        is_debug = self.args.debug
        return TrainingConfig(
            num_timesteps=5 if is_debug else 150_000_000,
            num_evals=15,
            reward_scaling=1.0,
            episode_length=1 if is_debug else self.env_config.episode_length,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=1 if is_debug else 32,
            num_updates_per_batch=1 if is_debug else 4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=0.005,
            num_envs=1 if is_debug else 8192,
            batch_size=2 if is_debug else 256,
            max_grad_norm=1.0,
            clipping_epsilon=0.2,
            num_resets_per_eval=1,
            network_factory=config_dict.create(
                policy_hidden_layer_sizes=(512, 256, 128),
                value_hidden_layer_sizes=(512, 256, 128),
                policy_obs_key="state",
                value_obs_key="privileged_state",
            ),
        )

    def _get_rl_config_dict(self) -> dict:
        config_dict = {k: v for k, v in asdict(self.training_config).items()}
        self.logger.info("RL config: %s", config_dict)
        return config_dict

    def save_video(self, frames: list[np.ndarray], fps: float, filename: str = "output.mp4") -> None:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

    def progress_callback(self, num_steps: int, metrics: dict) -> None:
        plt.figure()
        self.training_state.times.append(datetime.now())
        self.training_state.x_data.append(num_steps)
        self.training_state.y_data.append(metrics["eval/episode_reward"])
        self.training_state.y_dataerr.append(metrics["eval/episode_reward_std"])
        plt.xlim([0, self.training_state.rl_config["num_timesteps"] * 1.25])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={self.training_state.y_data[-1]:.3f}")
        plt.errorbar(
            self.training_state.x_data,
            self.training_state.y_data,
            yerr=self.training_state.y_dataerr,
            color="blue",
        )
        plt.savefig("plot.png")
        plt.close()

    def train(self) -> None:
        ppo_training_params = dict(self.training_state.rl_config)
        if "network_factory" in self.training_state.rl_config:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks,
                **self.training_state.rl_config.network_factory,
            )
            del ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks

        train_fn = functools.partial(
            ppo.train,
            **ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )

        self.logger.info("Time to jit: %s", self.training_state.times[1] - self.training_state.times[0])
        self.logger.info("Time to train: %s", self.training_state.times[-1] - self.training_state.times[1])

        if self.args.save_model:
            model.save_params(params, "params")

    def load_model(self) -> None:
        model_path = Path("checkpoints") / f"{self.env_name}_params.pkl"
        with open(model_path, "rb") as f:
            self.training_state.params = pickle.load(f)
        self.logger.info("Model loaded successfully")

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def run_eval_step(self, state: jax.Array, rng: jax.Array, inference_fn: any) -> tuple[jax.Array, jax.Array]:
        act_rng, next_rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        next_state = self.eval_env.step(state, ctrl)
        return next_state, next_rng

    def evaluate(self) -> None:
        """Evaluates the trained model by running episodes and optionally rendering them."""
        if self.training_state.params is None:
            self.load_model()

        # Create inference function
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **self.training_state.rl_config.network_factory,
        )
        policy_network = network_factory(self.eval_env.observation_size, self.eval_env.action_size)
        inference_fn = ppo_networks.make_inference_fn(policy_network)(self.training_state.params)
        inference_fn = jax.jit(inference_fn)

        # Run evaluation episodes
        for episode in range(self.training_config.num_resets_per_eval):
            rng = jax.random.PRNGKey(episode)
            state = self.eval_env.reset(rng)
            rollout = [state]
            modify_scene_fns = [lambda _: None]  # Default no-op scene modification

            # Run episode
            for _ in range(self.env_config.episode_length):
                state, rng = self.run_eval_step(state, rng, inference_fn)  # Pass inference_fn directly
                rollout.append(state)
                modify_scene_fns.append(lambda _: None)

            # Calculate episode statistics
            rewards = jnp.array([s.reward for s in rollout])
            episode_reward = jnp.sum(rewards)
            self.logger.info("Episode %d reward: %.2f", episode, episode_reward)

            # Render if requested
            self.render_episode(rollout, modify_scene_fns, episode)

    def render_episode(self, rollout: list[jax.Array], modify_scene_fns: list[callable], episode_num: int) -> None:
        render_every = 1
        fps = 1.0 / self.eval_env.dt / render_every
        self.logger.info("fps: %s", fps)

        traj = rollout[::render_every]
        mod_fns = modify_scene_fns[::render_every]

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

        frames = self.eval_env.render(
            traj,
            camera="track",
            scene_option=scene_option,
            width=640 * 2,
            height=480,
            modify_scene_fns=mod_fns,
        )

        self.save_video(frames, fps=fps, filename=f"output_{episode_num}.mp4")