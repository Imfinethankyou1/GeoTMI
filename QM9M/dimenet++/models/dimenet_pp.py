from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import (
    EmbeddingBlock,
    SphericalBasisLayer,
)
from torch_scatter import scatter
from torch_sparse import SparseTensor

from .bessel_basis_layer import BesselBasisLayer
from .interaction_block_pp import InteractionPPBlock
from .output_block_pp import OutputPPBlock


class DimeNetPlusPlus(Module):
    r"""DimeNet++ implementation based on https://github.com/klicperajo/dimenet.
    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size (int): Embedding size used in the basis transformation
        out_emb_channels(int): Embedding size used for atoms in the output block
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion.
            (default: :obj:`swish`)
    """

    def __init__(
        self,
        hidden_channels=128,
        out_channels=1,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
    ):
        super(DimeNetPlusPlus, self).__init__()

        self.cutoff = cutoff

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )
        self.act = act
        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = ModuleList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = ModuleList(
            [
                InteractionPPBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def triplets(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def get_angle(
        self, pos: Tensor, idx_i: Tensor, idx_j: Tensor, idx_k: Tensor
    ) -> Tensor:
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        pos_ji, pos_kj = (
            pos[idx_j].detach() - pos_i,
            pos[idx_k].detach() - pos_j,
        )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1) + 1e-12
        angle = torch.atan2(b, a)
        return angle

    def forward(
        self,
        z,
        pos,
        edge_index,
        batch: Optional[Batch] = None,
        update_pos: bool = False,
    ):
        j, i = edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0)
        )

        # calculate distance
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # calculate angles
        angle = self.get_angle(pos, idx_i, idx_j, idx_k)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # embedding block
        x = self.emb(z.long(), rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # interaction blocks
        dist_history = []
        for interaction_block, output_block, in zip(
            self.interaction_blocks,
            self.output_blocks[1:],
        ):
            x, pos_update_coeff = interaction_block(
                x, rbf, sbf, idx_kj, idx_ji, update_pos
            )
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

            if update_pos:
                pos_delta = scatter(
                    (pos[i] - pos[j]) * pos_update_coeff, i, dim=0, reduce="mean"
                )
                pos = pos + pos_delta

                dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
                angle = self.get_angle(pos, idx_i, idx_j, idx_k)

                rbf = self.rbf(dist)
                sbf = self.sbf(dist, angle, idx_kj)

                dist_history.append(dist)

        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
        return energy, dist_history
