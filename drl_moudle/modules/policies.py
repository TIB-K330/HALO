import copy
from abc import ABC, abstractmethod

import dgl
import numpy as np
from gym import spaces
import torch as th
import torch.nn as nn
from typing import List, Tuple, Type, Union

from crowd_sim.envs.policy.policy import Policy
from .type_aliases import GraphReplayBufferSamples, MaskedGraphReplayBufferSamples
from .torch_layers import (
    create_mlp,
    GraphExtractor,
)
from .utils import get_device


class MaskedGraphReplayBuffer(ABC):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """
    def __init__(
        self,
        buffer_size: int,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        handle_timeout_termination: bool = True,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.action_space = action_space
        self.device = get_device(device)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination

        if isinstance(action_space, spaces.Discrete):
            self.action_dim = 1  # Action is an int
            self.mask_dim = action_space.n
        else:
            raise NotImplementedError(f"{action_space} action space is not supported")


        self.observations = list()
        self.next_observations = list()
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.timeouts = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.action_masks = np.zeros((self.buffer_size, self.mask_dim), dtype=np.float32)
        self.next_action_masks = np.zeros((self.buffer_size, self.mask_dim), dtype=np.float32)


        self.pos = 0
        self.full = False

    def add(
        self,
        obs: dgl.DGLHeteroGraph,
        next_obs: dgl.DGLHeteroGraph,
        action: np.ndarray,
        reward: np.ndarray,
        done: bool,
        timeout: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        # Copy to avoid modification by reference
        if self.full:
            self.observations[self.pos] = copy.deepcopy(obs)
            self.next_observations[self.pos] = copy.deepcopy(next_obs)
        else:
            self.observations.append(copy.deepcopy(obs))
            self.next_observations.append(copy.deepcopy(next_obs))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.action_masks[self.pos] = np.array(action_mask).copy()
        self.next_action_masks[self.pos] = np.array(next_action_mask).copy()


        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([timeout])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> MaskedGraphReplayBufferSamples:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        data = (
            dgl.batch([self.observations[idx] for idx in batch_inds]).to(self.device),
            th.Tensor(self.actions[batch_inds]).to(self.device),
            dgl.batch([self.next_observations[idx] for idx in batch_inds]).to(self.device),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            th.Tensor(self.dones[batch_inds] * (1 - self.timeouts[batch_inds])).to(self.device),
            th.Tensor(self.rewards[batch_inds]).to(self.device),
            th.Tensor(self.action_masks[batch_inds]).to(self.device),
            th.Tensor(self.next_action_masks[batch_inds]).to(self.device),
        )
        return MaskedGraphReplayBufferSamples(*data)

class GraphReplayBuffer(ABC):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """
    def __init__(
        self,
        buffer_size: int,
        action_space: spaces.Box,
        device: Union[th.device, str] = "auto",
        handle_timeout_termination: bool = True,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.pos = 0
        self.full = False
        self.device = get_device(device)

        self.observations = list()
        self.next_observations = list()
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)

        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, 1), dtype=np.float32)

    def add(
        self,
        obs: dgl.DGLHeteroGraph,
        next_obs: dgl.DGLHeteroGraph,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        timeout: bool,
    ) -> None:
        # Copy to avoid modification by reference
        if self.full:
            self.observations[self.pos] = copy.deepcopy(obs)
            self.next_observations[self.pos] = copy.deepcopy(next_obs)
        else:
            self.observations.append(copy.deepcopy(obs))
            self.next_observations.append(copy.deepcopy(next_obs))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([timeout])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> GraphReplayBufferSamples:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        data = (
            dgl.batch([self.observations[idx] for idx in batch_inds]).to(self.device),
            th.Tensor(self.actions[batch_inds]).to(self.device),
            dgl.batch([self.next_observations[idx] for idx in batch_inds]).to(self.device),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            th.Tensor(self.dones[batch_inds] * (1 - self.timeouts[batch_inds])).to(self.device),
            th.Tensor(self.rewards[batch_inds]).to(self.device),
        )
        return GraphReplayBufferSamples(*data)


class GraphActor(nn.Module):
    """
    Actor network (policy) for TD3.

    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    """

    optimizer: th.optim.Optimizer
    features_extractor: GraphExtractor

    def __init__(
        self,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: GraphExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.net_arch = net_arch
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = int(np.prod(action_space.shape))
        actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        self.mu = nn.Sequential(*actor_net)

    def forward(self, obs: dgl.DGLHeteroGraph) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        return self.mu(features)

    def extract_features(self, obs: dgl.DGLHeteroGraph, features_extractor: GraphExtractor) -> th.Tensor:
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        cur_state = copy.deepcopy(obs)
        # exploration or test phase
        if cur_state.batch_size == 1:
            robot_state = cur_state.ndata['h'][0, 4:9].clone()
            return th.cat((robot_state, features_extractor(cur_state)[0, :]), dim=0)
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
            features = th.index_select(features_extractor(cur_state), 0, robot_ids)
            return th.cat((robot_states, features), dim=1)  # dim 0 is the batch dim, 1 is the feature dim

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)


class GraphContinuousCritic(nn.Module):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    optimizer: th.optim.Optimizer
    features_extractor: GraphExtractor

    def __init__(
        self,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: GraphExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__()
        action_dim = int(np.prod(action_space.shape))
        self.features_extractor = features_extractor
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: dgl.DGLHeteroGraph, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: dgl.DGLHeteroGraph, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](th.cat([features, actions], dim=1))

    def extract_features(self, obs: dgl.DGLHeteroGraph, features_extractor: GraphExtractor) -> th.Tensor:
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        cur_state = copy.deepcopy(obs)  # joint state as hetero-graph
        # exploration or test phase
        if cur_state.batch_size == 1:
            robot_state = cur_state.ndata['h'][0, 4:9].clone()
            return th.cat((robot_state, features_extractor(cur_state)[0, :]), dim=0)
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
            features = th.index_select(features_extractor(cur_state), 0, robot_ids)
            return th.cat((robot_states, features), dim=1)  # dim 0 is the batch dim, 1 is the feature dim

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)


class ActionNoise(ABC):
    """
    The action noise base class
    """

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        """
        Call end of episode reset for the noise
        """
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class ExternalPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'external'
        self.trainable = True
        self.multiagent_training = False,
        self.kinematics = 'differential'