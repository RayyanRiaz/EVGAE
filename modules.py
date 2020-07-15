from torch import nn
from torch_geometric.nn import GATConv


class FCNet(nn.Module):
    def __init__(self, sizes, last_layer_activation):
        super(FCNet, self).__init__()

        net = []
        for i in range(1, len(sizes)):
            net.append(nn.Linear(sizes[i - 1], sizes[i]))
            if i == len(sizes) - 1:
                if last_layer_activation is not None:
                    net.append(last_layer_activation)
            else:

                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class GCNNet(nn.Module):
    def __init__(self, sizes, layer_type=GATConv, last_layer_activation=None, **layer_args):
        super(GCNNet, self).__init__()
        self.layers = []
        for i in range(1, len(sizes)):
            self.layers.append(layer_type(in_channels=sizes[i - 1], out_channels=sizes[i], **layer_args))
        self.last_layer_activation = last_layer_activation

    def forward(self, x, edge_index):
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            if i == len(self.layers) - 1:
                if self.last_layer_activation is not None:
                    x = self.last_layer_activation(x)
            else:
                x = nn.ReLU(self.layers[i](x, edge_index))
