from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical, Normal
from gym import spaces

SelfDiagGaussianDistribution = TypeVar("SelfDiagGaussianDistribution", bound="DiagGaussianDistribution")
SelfSquashedDiagGaussianDistribution = TypeVar(
    "SelfSquashedDiagGaussianDistribution", bound="SquashedDiagGaussianDistribution"
)
SelfCategoricalDistribution = TypeVar("SelfCategoricalDistribution", bound="CategoricalDistribution")
SelfMultiCategoricalDistribution = TypeVar("SelfMultiCategoricalDistribution", bound="MultiCategoricalDistribution")


class CategoricalMasked(Categorical):
    def __init__(
        self,
        probs: Optional[th.Tensor] = None,
        logits: Optional[th.Tensor] = None,
        validate_args: Optional[bool] = None,
        masks: Optional[th.Tensor] = None
    ) -> None:
        self.masks = [] if masks is None else masks
        if len(self.masks) != 0:
            # self.masks = masks.type(th.bool).to(logits.device)
            self.masks = masks.to(logits.device)

            # logits = th.where(self.masks, logits, th.tensor(-1e+8).to(self.masks.device))
            logits = th.where(self.masks == 0.0, th.full_like(logits, -1e+8),
                              th.where(self.masks == 2.0, th.full_like(logits, 1e+8), logits))

        super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()  # without mask

        p_log_p = self.logits * self.probs
        # p_log_p = th.where(self.masks, p_log_p, th.tensor(0.).to(self.masks.device))
        en_masks = self.masks.type(th.bool).to(self.masks.device)
        p_log_p = th.where(en_masks, p_log_p, th.tensor(0.).to(self.masks.device))

        return -p_log_p.sum(-1)


class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super().__init__()
        self.distribution = None

    @abstractmethod
    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> th.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) -> th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(
        self: SelfDiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> SelfDiagGaussianDistribution:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """
    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None

    def proba_distribution(
        self: SelfSquashedDiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> SelfSquashedDiagGaussianDistribution:
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= th.sum(th.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return th.tanh(self.gaussian_actions)

    def mode(self) -> th.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return th.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(
        self: SelfCategoricalDistribution,
        action_logits: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> SelfCategoricalDistribution:
        # self.distribution = Categorical(logits=action_logits)
        self.distribution = CategoricalMasked(logits=action_logits, masks=action_masks)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(
        self,
        action_logits: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
        deterministic: bool = False
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits, action_masks)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        action_logits: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits, action_masks)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(
        self: SelfMultiCategoricalDistribution, action_logits: th.Tensor
    ) -> SelfMultiCategoricalDistribution:
        self.distribution = [
            Categorical(logits=split) for split in th.split(action_logits, tuple(self.action_dims), dim=1)
        ]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        return th.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


def make_proba_distribution(action_space: spaces.Space, dist_kwargs: Optional[Dict[str, Any]] = None) -> Distribution:
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        return DiagGaussianDistribution(int(np.prod(action_space.shape)))
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(action_space.n)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(list(action_space.nvec))
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )
