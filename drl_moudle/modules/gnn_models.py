# pylint: disable= no-member, arguments-differ, invalid-name
from typing import Optional
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch.linear import TypedLinear


class GNNFiLMLayer(nn.Module):
    """
    custom torch implemention of GNN-FiLM for crowd navigation
    """
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        num_rels: int,
        regularizer: Optional[str] = None,
        num_bases: Optional[str] = None,
        bias: bool = True,
        activation: Optional[nn.Module] = None,
        self_loop: bool = True,
        dropout: float = 0.0,
        layer_norm: bool = False
    ) -> None:
        super().__init__()
        if regularizer is not None and num_bases is None:
            num_bases = num_rels

        self.in_feat = in_feat
        self.out_feat = out_feat
        # self.etypes = ['h2r', 'o2r', 'w2r', 'o2h', 'w2h', 'h2h']
        self.bias = bias
        self.activation = activation
        self.relu = nn.ReLU()
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # weights for different types of edges
        self.W = TypedLinear(in_feat, out_feat, num_rels, regularizer, num_bases)

        # hypernets to learn the affine functions for different types of edges
        self.film = TypedLinear(in_feat, 2 * out_feat, num_rels, regularizer, num_bases)

        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def message_func(self, edges):
        m = self.W(edges.src['h'], edges.data['etype'], self.presorted)
        film_weights = self.film(edges.dst['h'], edges.data['etype'], self.presorted)
        gamma = film_weights[:, :self.out_feat]  # "gamma" for the affine function
        beta = film_weights[:, self.out_feat:]  # "beta" for the affine function
        m = gamma * m + beta  # compute messages
        m = self.relu(m)
        return {'m': m}

    def forward(self, graph, feat, etypes, presorted=False):
        # the input graph is a multi-relational graph, so treated as hetero-graph.
        self.presorted = presorted
        with graph.local_scope():
            graph.ndata['h'] = feat
            # graph.srcdata['h'] = feat
            # graph.dstdata['h'] = feat
            graph.edata['etype'] = etypes
            graph.update_all(self.message_func, fn.sum('m', 'h'))  # message passing
            # h = graph.dstdata['h']
            h = graph.ndata['h']
            if self.layer_norm:
                h = self.layer_norm_weight(h)

            if self.bias:
                h = h + self.h_bias

            if self.self_loop:
                h = h + feat[:graph.num_dst_nodes()] @ self.loop_weight

            if self.activation is not None:
                h = self.activation(h)

            h = self.dropout(h)
            return h
