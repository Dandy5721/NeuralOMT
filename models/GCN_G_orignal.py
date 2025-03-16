import torch.nn as nn
import torch.nn.functional as F
from .GCN_layers import GraphConvolution


# class GCN_G(nn.Module):
#     def __init__(self, infeature, nfeat, nhid, dropout):
#         super(GCN_G, self).__init__()
#         # self.linear = nn.Linear(nfeat, nfeat)
#         self.gc1 = GraphConvolution(nfeat, infeature)
#         self.norm1 = nn.LayerNorm(infeature)
#         # self.gc2 = GraphConvolution(infeature, nhid)
#         self.gc2 = GraphConvolution(infeature, 128)
#         self.norm2 = nn.LayerNorm(128)
#         self.linear2 = nn.Linear(128, 128)
#         self.gc3 = GraphConvolution(128, 256)
#         self.norm3 = nn.LayerNorm(256)
#         self.gc4 = GraphConvolution(256, 512)
#         self.norm4 = nn.LayerNorm(512)
#         self.linear4 = nn.Linear(512, 512)
#         self.gc5 = GraphConvolution(512, 256)
#         self.norm5 = nn.LayerNorm(256)
#         self.gc6 = GraphConvolution(256, 128)
#         self.norm6 = nn.LayerNorm(128)
#         # self.linear6 = nn.Linear(128, 128)
#         self.gc7 = GraphConvolution(128, 64)
#         self.norm7 = nn.LayerNorm(64)
#         self.linear7 = nn.Linear(64, 64)
#         self.gc8 = GraphConvolution(64, 32)
#         self.norm8 = nn.LayerNorm(32)
#
#         self.gc9 = GraphConvolution(32, 8)
#         self.norm9 = nn.LayerNorm(8)
#         self.linear = nn.Linear(8, 8)
#         self.gc10 = GraphConvolution(8, nhid)
#
#         self.dropout = dropout
#         # self.weight = nn.Parameter(torch.rand(infeature, infeature))
#
#         # self.weight = nn.Parameter(torch.ones(infeature, infeature))
#
#     def forward(self, x, adj, dis):
#         # print(x.shape)
#         # print(adj.shape)
#         # x_0 = F.relu(self.linear(x))
#         # print(x_0.shape)
#         # print('dddddddddd......', x.shape, adj.shape)
#         x_0 = F.relu(self.gc1(x, adj, dis))
#         print('-------------------x1--------------')
#         print(x_0)
#         # x_0 = self.norm1(x_0)
#         # x_0 = F.dropout(x_0, self.dropout)
#         # x_0 = x_0.float()
#         # f = F.tanh(self.linear(x_1))
#         # x = F.dropout(x, self.dropout, training=self.training)
#         x_0 = F.relu(self.gc2(x_0, adj, dis))
#         print('-------------------x2--------------')
#         print(x_0)
#         # x_0 = self.linear2(x_0)
#         # x_0 = self.norm2(x_0)
#
#         x_0 = F.relu(self.gc3(x_0, adj, dis))
#         print('-------------------x3--------------')
#         print(x_0)
#         # x_0 = self.norm3(x_0)
#         x_0 = F.relu(self.gc4(x_0, adj, dis))
#         # x_0 = self.linear4(x_0)
#         print('-------x4-----------')
#         print(x_0)
#         # x_0 = self.norm4(x_0)
#         # print('-------x4-norm----------')
#         # print(x_0)
#
#         x_0 = F.relu(self.gc5(x_0, adj, dis))
#         # x_0 = x + x_0
#
#         # res = x_0
#         # x_0 = self.norm5(x_0)
#         x_0 = F.relu(self.gc6(x_0, adj, dis))
#         # x_0 = self.norm6(x_0)
#         # x_0 = self.linear6(x_0)
#         x_0 = F.relu(self.gc7(x_0, adj, dis))
#         # x_0 = self.linear7(x_0)
#         # x_0 = self.norm7(x_0)
#
#         x_0 = F.relu(self.gc8(x_0, adj, dis))
#         # x_0 = self.norm8(x_0)
#         # x_0 = self.linear8(x_0)
#         x_0 = F.relu(self.gc9(x_0, adj, dis))
#         # print('-------x9-----------')
#         # print(x_0)
#         # x_0 = self.norm9(x_0)
#         # print('-------x9-norm----------')
#         # print(x_0)
#         # x_0 = self.linear(x_0)
#         x_0 = self.gc10(x_0, adj, dis)
#         # x_0 = x_0 + x
#         # print('-------x4-norm----------')
#         # print(x_0)
#
#         # x_1 = F.sigmoid(x_0)
#         return x_0




# class GCN_G(nn.Module):
#     def __init__(self, infeature, nfeat, nhid, dropout):
#         super(GCN_G, self).__init__()
#         # self.linear = nn.Linear(nfeat, infeature)
#         self.gc1 = GraphConvolution(nfeat, infeature)  #(nfeat, infeature)
#         self.norm1 = nn.LayerNorm(infeature)
#         # self.gc2 = GraphConvolution(infeature, nhid)
#         self.gc2 = GraphConvolution(infeature, 128)
#         self.norm2 = nn.LayerNorm(128)
#         self.linear2 = nn.Linear(128, 128)
#         self.gc3 = GraphConvolution(128, 64)
#         self.norm3 = nn.LayerNorm(64)
#         self.linear3 = nn.Linear(64, 64)
#         self.gc4 = GraphConvolution(64, 32)
#         self.linear4 = nn.Linear(32, 32)
#         self.norm4 = nn.LayerNorm(32)
#
#         self.gc5 = GraphConvolution(32, 8)
#         self.norm5 = nn.LayerNorm(8)
#         self.linear5 = nn.Linear(8, 8)
#         self.gc6 = GraphConvolution(8, nhid)
#
#         self.dropout = dropout
#         # self.weight = nn.Parameter(torch.rand(infeature, infeature))
#
#         # self.weight = nn.Parameter(torch.ones(infeature, infeature))
#
#     def forward(self, x, adj, dis):
#         # print(x.shape)
#         # print(adj.shape)
#         # x_0 = F.relu(self.linear(x))
#         # print(x_0.shape)
#         # print('dddddddddd......', x.shape, adj.shape)
#         # x_0 = self.linear(x)
#         x_0 = F.relu(self.gc1(x, adj, dis))
#         # print('-------------------x1--------------')
#         # print(x_0)
#         # x_0 = self.norm1(x_0)
#         # x_0 = F.dropout(x_0, self.dropout)
#         # x_0 = x_0.float()
#         # f = F.tanh(self.linear(x_1))
#         # x = F.dropout(x, self.dropout, training=self.training)
#         x_0 = F.relu(self.gc2(x_0, adj, dis))
#         # print('-------------------x2--------------')
#         x_0 = self.linear2(x_0)
#         # print(x_0)
#         # x_0 = self.linear2(x_0)
#         # x_0 = self.norm2(x_0)
#
#         x_0 = F.relu(self.gc3(x_0, adj, dis))
#         # print('-------------------x3--------------')
#         # print(x_0)
#         # x_0 = self.norm3(x_0)
#         x_0 = F.relu(self.gc4(x_0, adj, dis))
#         x_0 = self.linear4(x_0)
#         # print('-------x4-----------')
#         # print(x_0)
#         # x_0 = self.norm4(x_0)
#         # print('-------x4-norm----------')
#         # print(x_0)
#
#         x_0 = F.relu(self.gc5(x_0, adj, dis))
#         # x_0 = x + x_0
#
#         # res = x_0
#         # x_0 = self.norm5(x_0)
#         x_0 = self.gc6(x_0, adj, dis)
#
#         return x_0


# class GCN_G(nn.Module):
#     def __init__(self, infeature, nfeat, nhid, dropout, norm=None):
#         super(GCN_G, self).__init__()
#         self.linear0 = nn.Linear(nfeat, nfeat)
#         self.gc0 = GraphConvolution(nfeat, infeature)
#
#         self.linear1 = nn.Linear(infeature, infeature)
#         self.gc1 = GraphConvolution(infeature, infeature)
#
#         self.linear_med1 = nn.Linear(infeature, infeature)
#         self.linear_med2 = nn.Linear(infeature, infeature)
#         self.linear_med3 = nn.Linear(infeature, infeature)
#         self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)
#         self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)
#         self.leakyrelu3 = nn.LeakyReLU(0.2, inplace=True)
#         self.linear_med4 = nn.Linear(infeature, infeature)
#
#         # add relu
#         gconv_hidden = []
#         hidden_layer_count = 1
#         for _ in range(hidden_layer_count):
#             # No weighted edges and no propagated coordinates in hidden layers
#             gc_layer = GraphConvolution(infeature, infeature)
#             # if norm == 'batch':
#             #     norm_layer = nn.BatchNorm1d(infeature)
#             # elif norm == 'layer':
#             #     norm_layer = nn.LayerNorm(infeature)
#             gconv_hidden += [gc_layer]
#         self.gconv_hidden = nn.Sequential(*gconv_hidden)
#
#         # add relu
#         self.linear2 = nn.Linear(infeature, infeature)
#         self.gc2 = GraphConvolution(infeature, infeature)
#         self.linear3 = nn.Linear(infeature, nhid)
#         # add relu
#
#     def forward(self, x, adj, dis):
#         x_0 = self.gc0(self.linear0(x), adj, dis)
#
#         x_0 = self.gc1(self.linear1(x_0), adj, dis)
#
#         x_0 = self.leakyrelu1(self.linear_med1(x_0))
#         x_0 = self.leakyrelu2(self.linear_med2(x_0))
#         x_0 = self.leakyrelu3(self.linear_med3(x_0))
#         x_0 = F.relu(self.linear_med4(x_0))
#
#         x_res = x_0
#         for i, gconv in enumerate(self.gconv_hidden):
#             x_0 = F.relu(gconv(x_0, adj, dis) + x_res )  #  x_0.detach()
#
#         x_0 = self.linear2(x_0)
#         x_0 = self.gc2(x_0, adj, dis)
#         x_0 = self.linear3(x_0)
#
#         return x_0

class GCN_G(nn.Module):
    def __init__(self, infeature, nfeat, nhid, dropout):
        super(GCN_G, self).__init__()
        # self.linear = nn.Linear(nfeat, nfeat)
        self.gc1 = GraphConvolution(nfeat, infeature)
        # self.gc2 = GraphConvolution(infeature, 64)
        self.gc2 = GraphConvolution(infeature, 128)
        self.gc3 = GraphConvolution(128, 86)
        self.gc4 = GraphConvolution(86, 64)
        self.gc5 = GraphConvolution(64, 32)
        self.gc6 = GraphConvolution(32, 16)
        self.gc7 = GraphConvolution(16, 8)
        self.gc8 = GraphConvolution(8, nhid)
        self.dropout = dropout

    def forward(self, x, adj, dis):
        # print(x.shape)
        # print(adj.shape)
        # x_0 = F.relu(self.linear(x))
        # print(x_0.shape)
        # print(x.dtype)
        # print(adj.dtype)
        x_0 = F.relu(self.gc1(x, adj, dis))
        # x_0 = F.dropout(x_0, self.dropout)
        # x_0 = x_0.float()
        # f = F.tanh(self.linear(x_1))
        # x = F.dropout(x, self.dropout, training=self.training)
        x_0 = F.relu(self.gc2(x_0, adj, dis))
        x_0 = F.relu(self.gc3(x_0, adj, dis))
        x_0 = F.relu(self.gc4(x_0, adj, dis))
        x_0 = F.relu(self.gc5(x_0, adj, dis))
        x_0 = F.relu(self.gc6(x_0, adj, dis))
        x_0 = F.relu(self.gc7(x_0, adj, dis))
        x_0 = self.gc8(x_0, adj, dis)
        # x_1 = F.log_softmax(x_0, dim=0)
        return x_0
        # return F.log_softmax(x, dim=1)