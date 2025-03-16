import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from pytorch3d.ops import GraphConv



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.gc = GraphConv(out_features, out_features)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            # nn.init.xavier_uniform_(self.bias)

    def forward(self, input, adj, dis):
        # print('init param', 1. / math.sqrt(self.weight.size(1)))
        device = input.device
        support = torch.mm(input, self.weight)
        # support = support.double()
        # adj = adj.float()
        # output = self.gc(support, adj)
        # num_vert = support.size(0)
        zeros_m = torch.zeros((1, support.size(1))).to(device)

        add_support = torch.cat((support, zeros_m), dim=0)
        # print("shapeeee", add_support[adj.long(),:].shape)
        # print('ddddd', dis[adj[:,0].long(),:].shape)
        output = (add_support[adj.long(),:] * dis[adj[:,0].long(),:].unsqueeze(2)).sum(1)
        # for i in range(input.size(0)):
        #     index = torch.tensor(adj[i])
        #     tmp = input[index].sum(0) + input[i]
        #     output[i,:] = tmp

        # output = torch.spmm(adj, support) #change
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
