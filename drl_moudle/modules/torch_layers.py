from typing import Dict, List, Tuple, Type, Union
import torch
from torch import nn

import dgl
from dgl.nn.pytorch.conv.gatconv import GATConv

from .gnn_models import GNNFiLMLayer


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class MlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: torch.device
    ) -> None:
        super().__init__()
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class GraphExtractor(nn.Module):
    def __init__(
        self,
        obs_dim: int = 44,
        features_dim: int = 64,
        num_rels: int = 6,
        target_drop_rate: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        assert features_dim > 0
        self._features_dim = features_dim
        self.encoder_dim = features_dim
        self.encoder = nn.Sequential(*create_mlp(obs_dim, self.encoder_dim, []))

        # self.gnn = GNNFiLMLayer(
        #     self.encoder_dim,
        #     self._features_dim,
        #     num_rels,
        #     dropout=target_drop_rate,
        #     layer_norm=layer_norm
        # )

        self.gnn = GATConv(self.encoder_dim, self._features_dim, num_heads=1, activation=torch.nn.ReLU())

        # self.gnn_layer1 = GATConv(self.encoder_dim, self._features_dim, num_heads=1, activation=torch.nn.ReLU())
        # self.dense_layer = nn.Sequential(*create_mlp(self._features_dim, self._features_dim, []))
        # self.gnn_layer2 = GATConv(self._features_dim, self._features_dim, num_heads=1, activation=torch.nn.ReLU())

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: dgl.DGLHeteroGraph):
        sub_graph = observations  #  shallow copy
        node_features = observations.ndata['h']  # h0
        edge_types = observations.edata['rel_type']
        sub_graph.set_n_initializer(dgl.init.zero_initializer)
        sub_graph.set_e_initializer(dgl.init.zero_initializer)

        z1 = self.encoder(node_features)
        # h1 = self.gnn(sub_graph, z1, edge_types)
        sub_graph = dgl.add_self_loop(sub_graph)
        h1 = self.gnn(sub_graph, z1)
        h1 = h1.reshape(-1, self._features_dim)
        return h1 + z1

        # z1 = self.encoder(node_features)
        # sub_graph = dgl.add_self_loop(sub_graph)
        # h1 = self.gnn_layer1(sub_graph, z1)
        # h1 = h1.reshape(-1, self._features_dim)
        # z2 = self.dense_layer(h1)
        # h2 = self.gnn_layer2(sub_graph, z2)
        # h2 = h2.reshape(-1, self._features_dim)
        # return  h1 + h2


def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch
