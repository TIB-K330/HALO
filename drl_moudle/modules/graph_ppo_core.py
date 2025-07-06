import copy
import pdb
import warnings
import gym
from gym import spaces
import numpy as np
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Type, Union
from abc import ABC, abstractmethod
from functools import partial

import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import dgl

from .torch_layers import MlpExtractor, GraphExtractor
from .distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    make_proba_distribution,
    MultiCategoricalDistribution,
)
from .type_aliases import Schedule, GraphRolloutBufferSamples
from .utils import get_device


class GraphRolloutBuffer(ABC):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
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

        if isinstance(action_space, spaces.Box):
            self.action_dim = int(np.prod(action_space.shape))
        elif isinstance(action_space, spaces.Discrete):
            self.action_dim = 1  # Action is an int
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = int(len(action_space.nvec))
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
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
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
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[GraphRolloutBufferSamples, None, None]:
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
    ) -> GraphRolloutBufferSamples:

        data = (
            dgl.batch([self.observations[idx] for idx in batch_inds]).to(self.device),
            th.Tensor(self.actions[batch_inds]).to(self.device),
            th.Tensor(self.values[batch_inds]).to(self.device),
            th.Tensor(self.log_probs[batch_inds]).to(self.device),
            th.Tensor(self.advantages[batch_inds]).to(self.device),
            th.Tensor(self.returns[batch_inds]).to(self.device),
        )
        return GraphRolloutBufferSamples(*data)


class GraphActorCriticPolicy(nn.Module):

    optimizer: th.optim.Optimizer
    features_extractor: GraphExtractor

    def __init__(
        self,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        features_extractor_class: Type[GraphExtractor] = GraphExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        if net_arch is None:
            net_arch = dict(pi=[256, 256], vf=[256, 256])

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.action_space = action_space
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim + 5  # concat with robot state

        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.log_std_init = log_std_init
        self.action_dist = make_proba_distribution(action_space)
        self._build(lr_schedule)
        self.actions_probs: th.Tensor = None

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def make_features_extractor(self):
        return self.features_extractor_class(**self.features_extractor_kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            module_gains = {
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

            # TODO: Orthogonal initialize graph extractor

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @property
    def device(self) -> th.device:
        for param in self.parameters():
            return param.device
        return th.device("cpu")

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)  # nn.Module.train

    def forward(
        self,
        state: dgl.DGLHeteroGraph,  # joint state
        deterministic: bool = False,
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
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def extract_features(self, obs: dgl.DGLHeteroGraph) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        cur_state = copy.deepcopy(obs)  # features_extractor will disrupt the original features
        # exploration or test phase
        if obs.batch_size == 1:
            robot_state = cur_state.ndata['h'][0, 4:9].clone()
            if self.share_features_extractor:
                return th.cat((robot_state, self.features_extractor(cur_state)[0, :]), dim=0)
            else:
                cur_state1 = copy.deepcopy(obs)
                pi_features = self.pi_features_extractor(cur_state)[0, :]
                pi_features = th.cat((robot_state, pi_features), dim=0)
                vf_features = self.vf_features_extractor(cur_state1)[0, :]
                vf_features = th.cat((robot_state, vf_features), dim=0)
                return pi_features, vf_features
        # batch training phase
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

            if self.share_features_extractor:
                shared_features = th.index_select(self.features_extractor(cur_state), 0, robot_ids)
                return th.cat((robot_states, shared_features), dim=1)
            else:
                cur_state1 = copy.deepcopy(obs)
                pi_features = th.index_select(self.pi_features_extractor(cur_state), 0, robot_ids)
                pi_features = th.cat((robot_states, pi_features), dim=1)  # dim 0 is the batch dim, 1 is the feature dim
                vf_features = th.index_select(self.vf_features_extractor(cur_state1), 0, robot_ids)
                vf_features = th.cat((robot_states, vf_features), dim=1)
                return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def predict(self, observation: dgl.DGLHeteroGraph, deterministic: bool = False) -> np.ndarray:
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        with th.no_grad():
            actions = self.get_distribution(observation).get_actions(deterministic=deterministic)

        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))
        if isinstance(self.action_space, spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return actions

    def evaluate_actions(
        self,
        obs: dgl.DGLHeteroGraph,
        actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
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

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: dgl.DGLHeteroGraph) -> Distribution:
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
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: dgl.DGLHeteroGraph) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        cur_state = copy.deepcopy(obs)
        if obs.batch_size == 1:
            robot_state = cur_state.ndata['h'][0, 4:9].clone()
            features = self.vf_features_extractor(cur_state)[0, :]
            features = th.cat((robot_state, features), dim=0)
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

            features = th.index_select(self.vf_features_extractor(cur_state), 0, robot_ids)
            features = th.cat((robot_states, features), dim=1)

        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
