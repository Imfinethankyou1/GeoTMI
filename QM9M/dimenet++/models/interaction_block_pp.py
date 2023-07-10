from typing import Callable

import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, Sequential
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import ResidualLayer
from torch_scatter import scatter


class Swish(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor):
        return x * torch.sigmoid(x)


class InteractionPPBlock(Module):
    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )

        # legacy
        # self.pos_update_mlp = Sequential(
        #     Linear(hidden_channels, hidden_channels * 2),
        #     Swish(),
        #     Linear(hidden_channels * 2, 1),
        # )
        self.rbf_att1 = Linear(hidden_channels, hidden_channels, bias=True)
        self.rbf_att2 = Linear(hidden_channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.rbf_att1.weight, scale=2.0)
        glorot_orthogonal(self.rbf_att2.weight, scale=2.0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()

        # for param in self.pos_update_mlp.parameters():
        #     if param.dim() > 1:
        #         glorot_orthogonal(param, scale=2.0)
        #     else:
        #         param.data.fill_(0)

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        sbf: Tensor,
        idx_kj: Tensor,
        idx_ji: Tensor,
        update_pos: bool = False,
    ) -> Tensor:
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        # pos_update_coeff = None
        # if update_pos:
        #     pos_update_coeff = torch.tanh(self.pos_update_mlp(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        pos_update_coeff = None
        if update_pos:
            pos_update_coeff = self.act(self.rbf_att1(h))
            pos_update_coeff = torch.tanh(self.rbf_att2(pos_update_coeff))

        return h, pos_update_coeff
