from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Type, Union
from abc import ABC
import copy
import dgl
from gym import spaces
import numpy as np
import torch as th
import torch.nn as nn

from .distributions import CategoricalDistribution, Distribution
from .graph_ppo_core import GraphActorCriticPolicy
from .torch_layers import GraphExtractor
from .type_aliases import MaskedRolloutBufferSamples, Schedule
from .utils import get_device


class MaskedRolloutBuffer(ABC):
    """
    Masked Rollout buffer used in MpcPPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    """
    observations: List[dgl.DGLHeteroGraph]
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    action_masks: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.buffer_size = buffer_size

        if isinstance(action_space, spaces.Discrete):
            self.action_dim = 1  # Action is an int
            self.mask_dim = action_space.n
        else:
            raise NotImplementedError(f"{action_space} action space is not supported")

        self.pos = 0
        self.full = False
        self.device = get_device(device)

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = list()
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.action_masks = np.zeros((self.buffer_size, self.mask_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            # if next_non_terminal is false, next_values or last_gae_lam no longer belong to this episode,
            # so interrupt the chain.
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see GitHub PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: dgl.DGLHeteroGraph,
        action: np.ndarray,
        reward: float,
        episode_start: bool,
        value: th.Tensor,
        log_prob: th.Tensor,
        mask: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param mask: action mask
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations.append(copy.deepcopy(obs))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.action_dim,))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.action_masks[self.pos] = mask.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[MaskedRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)

        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> MaskedRolloutBufferSamples:

        data = (
            dgl.batch([self.observations[idx] for idx in batch_inds]).to(self.device),
            th.Tensor(self.actions[batch_inds]).to(self.device),
            th.Tensor(self.values[batch_inds]).to(self.device),
            th.Tensor(self.log_probs[batch_inds]).to(self.device),
            th.Tensor(self.advantages[batch_inds]).to(self.device),
            th.Tensor(self.returns[batch_inds]).to(self.device),
            th.Tensor(self.action_masks[batch_inds]).to(self.device),
        )
        return MaskedRolloutBufferSamples(*data)


class MaskedActorCriticPolicy(GraphActorCriticPolicy):

    def forward(
        self,
        state: dgl.DGLHeteroGraph,  # joint state
        deterministic: bool = False,
        action_mask: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Preprocess the observation if needed
        features = self.extract_features(state)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, action_masks=action_mask)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        self.actions_probs = distribution.distribution.probs
        return actions, values, log_prob

    def _get_action_dist_from_latent(
        self,
        latent_pi: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions, action_masks=action_masks)
        else:
            raise ValueError("Invalid action distribution")

    def predict(
        self,
        observation: dgl.DGLHeteroGraph,
        deterministic: bool = False,
        action_mask: Optional[th.Tensor] = None,
    ) -> np.ndarray:
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        with th.no_grad():
            # actions = self.get_distribution(observation, action_mask).get_actions(deterministic=deterministic)
            distribution = self.get_distribution(observation, action_mask)
            actions = distribution.get_actions(deterministic=deterministic)
            self.actions_probs = distribution.distribution.probs
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        return actions

    def evaluate_actions(
        self,
        obs: dgl.DGLHeteroGraph,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :param action_masks: by Alex
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi, action_masks=action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: dgl.DGLHeteroGraph, action_masks: Optional[th.Tensor] = None) -> Distribution:
        cur_state = copy.deepcopy(obs)
        if obs.batch_size == 1:
            robot_state = cur_state.ndata['h'][0, 4:9].clone()
            features = self.pi_features_extractor(cur_state)[0, :]
            features = th.cat((robot_state, features), dim=0)
            features = features.unsqueeze(0)  # add batch dim
        else:
            batch_features = cur_state.ndata['h']
            num_nodes = cur_state._batch_num_nodes['_N'].cpu().numpy()
            robot_ids = []
            robot_id = 0
            for i in range(num_nodes.shape[0]):
                robot_ids.append(robot_id)
                robot_id += num_nodes[i]

            robot_ids = th.LongTensor(robot_ids).to(self.device)
            robot_features = th.index_select(batch_features, 0, robot_ids)
            robot_states = robot_features[:, 4:9].clone()

            features = th.index_select(self.pi_features_extractor(cur_state), 0, robot_ids)
            features = th.cat((robot_states, features), dim=1)

        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi, action_masks=action_masks)
