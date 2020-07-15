import math
import random

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.distributions import kl_divergence, Categorical
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.utils import to_undirected, negative_sampling

from modules import FCNet


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logvar = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))

        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class QyGivenX(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(QyGivenX, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return torch.softmax(self.conv2(x, edge_index), 1)


class EpitomeVGAE(VGAE):
    def __init__(self, x_dim, z_dim, encoder=None, edge_decoder=None, node_decoder=None, num_epitomes=4, shifts=0):
        if encoder is None:
            enc = Encoder(in_channels=x_dim, out_channels=z_dim)
        else:
            enc = encoder
        super(EpitomeVGAE, self).__init__(enc, edge_decoder)
        if node_decoder:
            self.node_decoder = node_decoder
        else:
            self.node_decoder = FCNet(sizes=[z_dim, 10 * z_dim, 5 * z_dim, x_dim], last_layer_activation=nn.Sigmoid())

        self.z_dim = z_dim

        self.epitomes = nn.Parameter(torch.eye(num_epitomes).repeat_interleave(int(z_dim / num_epitomes), 1), requires_grad=False)
        self.epitomes.data = self.epitomes + torch.roll(self.epitomes, shifts=shifts, dims=1)
        self.epitomes.data = (self.epitomes > 0).float()
        # self.epitomes = nn.Parameter(((1 - torch.eye(num_epitomes)) * 1e-0 + torch.eye(num_epitomes)).repeat_interleave(int(z_dim / num_epitomes), 1),
        #                              requires_grad=False)
        # TODO try with a GCN here too!
        self.qy_given_x = FCNet(sizes=([x_dim, z_dim * 2, num_epitomes]), last_layer_activation=nn.Softmax(dim=1))


    def kl_losses(self, x, edge_index):
        l_kl = -0.5 * (1 + self.__logvar__ - self.__mu__ ** 2 - self.__logvar__.exp())

        qy_given_x = self.qy_given_x(x)
        l_kl = (qy_given_x[:, :, None] * self.epitomes[None, :, :] * l_kl[:, None, :]).sum(dim=1)
        l_q = kl_divergence(Categorical(qy_given_x), Categorical(torch.ones_like(qy_given_x)))
        return l_kl, l_q

    def test_model(self, train_z, train_y, test_z, test_y, solver='lbfgs', multi_class='auto', *args, **kwargs):

        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args, **kwargs).fit(
            train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())

        return clf.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy())

    @staticmethod
    def split_edges(data, val_ratio=0.05, test_ratio=0.1):
        r""" Copied from pytorch 1.4 - Splits the edges of a :obj:`torch_geometric.data.Data` object
        into positve and negative train/val/test edges.

        Args:
            data (Data): The data object.
            val_ratio (float, optional): The ratio of positive validation
                edges. (default: :obj:`0.05`)
            test_ratio (float, optional): The ratio of positive test
                edges. (default: :obj:`0.1`)
        """

        assert 'batch' not in data  # No batch-mode.

        row, col = data.edge_index
        data.edge_index = None

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        # Negative edges.
        num_nodes = data.num_nodes
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero().t()
        perm = random.sample(range(neg_row.size(0)),
                             min(n_v + n_t, neg_row.size(0)))
        perm = torch.tensor(perm)
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

        return data


    def recon_loss_without_reduction(self, z, pos_edge_index, neg_edge_index=None):

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15)

        return pos_loss, neg_loss
