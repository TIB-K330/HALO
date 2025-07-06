"""Common aliases for type hints"""
from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Union

import numpy as np
import torch as th
import dgl

from .callbacks import BaseCallback

TensorDict = Dict[str, th.Tensor]
MaybeCallback = Union[None, Callable, List[BaseCallback], BaseCallback]

# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]


class GraphRolloutBufferSamples(NamedTuple):
    observations: dgl.DGLHeteroGraph
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class MaskedRolloutBufferSamples(NamedTuple):
    observations: dgl.DGLHeteroGraph
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor


class GraphReplayBufferSamples(NamedTuple):
    observations: dgl.DGLHeteroGraph
    actions: th.Tensor
    next_observations: dgl.DGLHeteroGraph
    dones: th.Tensor
    rewards: th.Tensor

class MaskedGraphReplayBufferSamples(NamedTuple):
    observations: dgl.DGLHeteroGraph
    actions: th.Tensor
    next_observations: dgl.DGLHeteroGraph
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor
    next_action_masks: th.Tensor


class RolloutReturn(NamedTuple):
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class TrainFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class TrainFreq(NamedTuple):
    frequency: int
    unit: TrainFrequencyUnit  # either "step" or "episode"
