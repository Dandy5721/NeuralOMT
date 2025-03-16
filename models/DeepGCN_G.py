import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import DeepGCNLayer, GENConv

class DeeperGCN_G(torch.nn.Module):
    def __init__(self, infeatures, hidden_channels = 64, num_layers = 1, nid = 1):
        super().__init__()
        torch.manual_seed(12345)
        # self._set_static_graph()
        self.node_encoder = nn.Linear(infeatures, hidden_channels)
        self.Norm = nn.LayerNorm(hidden_channels * (2 ** num_layers), elementwise_affine=True)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels * (2 ** (i - 1)), hidden_channels * (2 ** (i - 1)), aggr='softmax', t=1.0,
                           learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels * (2 ** (i - 1)), elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='dense', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = nn.Linear(hidden_channels * (2 ** num_layers), nid)

    def forward(self, x, edge_index):
        x = self.node_encoder(x)

        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.Norm(x)
        x = self.layers[0].act(x)
        # x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)