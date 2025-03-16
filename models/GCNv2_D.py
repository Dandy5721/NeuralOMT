import torch.nn as nn
import torch.nn.functional as F
from GCN_layers import GraphConvolution
import torch
from torch_geometric.nn import GCN2Conv
class GCNv2_D(torch.nn.Module):
    def __init__(self, infeatures, num_layers, alpha = 0.1, theta= 0.5, nhid = 1,
                 shared_weights=True, dropout=0.0):
        super().__init__()
        torch.manual_seed(12345)
        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(infeatures, 64))
        self.lins.append(nn.Linear(64, nhid))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(64, alpha, theta, layer + 1, shared_weights, normalize = False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x