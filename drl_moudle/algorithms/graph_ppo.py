import sys
import time
from collections import deque
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import random

import numpy as np
import torch as th
from torch.nn import functional as F
import dgl
import gym
from modules.type_aliases import MaybeCallback, Schedule
from modules.base_class import BaseAlgorithm
from modules.callbacks import BaseCallback
from modules.graph_ppo_core import GraphActorCriticPolicy, GraphRolloutBuffer
from modules.utils import (
    configure_logger,
    explained_variance,
    get_schedule_fn,
    joint_state_as_graph,
    safe_mean,
)
from crowd_sim.envs.utils.action import ActionDiff
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import map_action_to_accel

SelfGraphPPO = TypeVar("SelfGraphPPO", bound="GraphPPO")


class GraphPPO(BaseAlgorithm):
    """
    :param policy: The policy model to use (GraphPolicy)
    :param env: The environment to learn from (if registered in Gym, can be str)
    """

    policy_aliases: ClassVar[Dict[str, Any]] = {"GraphPolicy": GraphActorCriticPolicy, }

    def __init__(
        self,
        policy: Union[str, Type[GraphActorCriticPolicy]],
        env: gym.Env,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Union[None, float] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        target_kl: Optional[float] = None,
        policy_kwargs: Optional[Dict[str, any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed
        )

        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        self._last_episode_start = None
        # Track the training progress remaining (from 1 to 0), for clip range updating
        self._current_progress_remaining = 1.0

        # Hyper parameters
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self._num_episodes = 0
        self.rewards: List[float] = []  # adopt from 'monitor.py' in original SB3

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        self.set_random_seed(self.seed)
        self.rollout_buffer = GraphRolloutBuffer(
            self.n_steps,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        # The observation of crowd_sim is not compliance with the gym standard
        self.policy = self.policy_class(self.action_space, self.lr_schedule, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: gym.Env,
        callback: BaseCallback,
        rollout_buffer: GraphRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_state is not None, "No previous state graph was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                actions, values, log_probs = self.policy(self._last_state)

            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, gym.spaces.Box):
                # Clip the actions to avoid out of bound error as we are sampling from an unbounded Gaussian distribution
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            clipped_action = clipped_actions.squeeze()  # remove the batch dim

            if isinstance(self.action_space, gym.spaces.Discrete):
                left_acc, right_acc = map_action_to_accel(clipped_action)
                new_ob, reward, done, info = env.step(ActionDiff(left_acc, right_acc))
            else:
                new_ob, reward, done, info = env.step(ActionDiff(clipped_action[0], clipped_action[1]))

            new_robot_state = env.robot.get_full_state()
            env.render(mode='debug')

            self.num_timesteps += 1

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self.rewards.append(float(reward))  # for return calculation
            if done:
                returns = []
                for step in range(len(self.rewards)):
                    step_return = sum([pow(self.gamma, t) * reward for t, reward in enumerate(self.rewards[step:])])
                    returns.append(step_return)

                ep_info = {"r": round(safe_mean(returns), 6), "l": len(self.rewards)}
                self.rewards = []

                self.ep_info_buffer.append(ep_info)
                self.ep_success_buffer.append(True if isinstance(info, ReachGoal) else False)

            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            if done:
                if isinstance(info, Timeout):
                    terminal_state = joint_state_as_graph(JointState(new_robot_state, new_ob), device=self.device)
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_state).cpu().numpy()[0]
                    reward += self.gamma * terminal_value

                self._episode_num += 1

                new_ob = env.reset(phase="train")
                new_robot_state = env.robot.get_full_state()

            rollout_buffer.add(
                self._last_state,
                actions,
                reward,
                self._last_episode_start,
                values,
                log_probs,
            )
            self._last_state = joint_state_as_graph(JointState(new_robot_state, new_ob), device=self.device)
            self._last_episode_start = done

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(self._last_state)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.policy.set_training_mode(True)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        # save data for logging or visualization
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True  # early stop flag
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            total_timesteps += self.num_timesteps  # Make sure training timesteps are ahead of the internal counter

        if reset_num_timesteps or self._last_state is None:
            assert self.env is not None
            ob = self.env.reset(phase="train")
            robot_state = self.env.robot.get_full_state()
            self._last_state = joint_state_as_graph(JointState(robot_state, ob), device=self.device)
            self._last_episode_start = True  # Original SB3 is a ndarray, for multiple environment

        if not self._custom_logger:
            self._logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
        # Create eval callback if needed
        callback = self._init_callback(callback)
        return total_timesteps, callback

    def learn(
        self: SelfGraphPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
    ) -> SelfGraphPPO:
        iteration = 0
        total_timesteps, callback = self._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name)
        callback.on_training_start(locals(), globals())
        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)  # for clip_range update

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                # time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/ep_ret_mean",
                        safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                    )
                    self.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    )
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()
        return self

    def predict(self, observation: JointState, deterministic: bool = False) -> np.ndarray:
        ob_graph = joint_state_as_graph(observation, device=self.device)
        return self.policy.predict(ob_graph, deterministic)

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this, so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "rollout_buffer",
            "_logger",
            "_custom_logger",
            "ocp_planner",
            "ocp_planner_thread",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []
