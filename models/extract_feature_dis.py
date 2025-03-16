
from pytorch3d.ops import GraphConv 
from pytorch3d.structures import Meshes
from .graph_conv import Features2FeaturesResidual, GraphConvNorm
import torch
from collections import defaultdict

# class SurfNN(nn.Module):
#     def __init__(self, n_in_channels=1, n_start_filters=16):
#         super(SurfNN, self).__init__()
#
#         self.graph_conv_first = Features2FeaturesResidual(3, n_start_filters, hidden_layer_count=2, norm='batch', GC=GraphConvNorm, weighted_edges=False)
#
#         self.gc1 = Features2FeaturesResidual(n_start_filters * 2, n_start_filters, hidden_layer_count=2, norm='batch', GC=GraphConvNorm, weighted_edges=False)
#         self.gc1_f2v = GraphConvNorm(n_start_filters, 3, weighted_edges=False,init='zero')
#
#
#     def forward(self, v, f, volume, n_smooth=1, lambd=1.0):
#
#         # the first gc layer
#         temp_meshes = Meshes(verts=v, faces=f)
#         edges_packed = temp_meshes.edges_packed()
#         verts_packed = temp_meshes.verts_packed()
#
#         xxx = self.graph_conv_first(verts_packed, edges_packed).unsqueeze(0)
#
#         return xxxx

    # def initialize(self, L, W, H, device=None):
    #     self.block.initialize(L, W, H, device)
    #
    #     layer_list1 = []
    #     layer_list2 = []
    #     layer_list1.append(self.graph_conv_first)
    #     layer_list1.append(self.gc1)
    #     layer_list2.append(self.gc1_f2v)
    #
    #     for m in layer_list1:
    #         if isinstance(m, GraphConv):
    #             nn.init.xavier_uniform_(m.lin.weight)
    #             nn.init.constant_(m.lin.bias, 0)
    #     for m in layer_list2:
    #         if isinstance(m, GraphConv):
    #             nn.init.constant_(m.w0.weight, 0.0)
    #             nn.init.constant_(m.w0.bias, 0.0)
    #             nn.init.constant_(m.w1.weight, 0.0)
    #             nn.init.constant_(m.w1.bias, 0.0)



class SurfNN(torch.nn.Module):
    def __init__(self, n_in_channels=3, n_start_filters=1): #n_start_filters=3
        super(SurfNN, self).__init__()

        self.graph_conv_first = Features2FeaturesResidual(1, n_start_filters, hidden_layer_count=2, norm='layer',
                                                          GC=GraphConvNorm, weighted_edges=False)
        self.gc1 = Features2FeaturesResidual(n_start_filters, n_start_filters, hidden_layer_count=2, norm='layer',
                                             GC=GraphConvNorm, weighted_edges=False)
        self.gc1_f2v = GraphConvNorm(n_start_filters, 1, weighted_edges=False, init='zero')

    def forward(self, mgh, f, v, n_smooth=1, lambd=1.0):
        device = v.device
        temp_meshes = Meshes(verts=v, faces=f)
        edges_packed = temp_meshes.edges_packed()
        verts_packed = temp_meshes.verts_packed()


        # features = self.gc1(features, edges_packed)
        # features = self.gc1_f2v(features, edges_packed)

        # print('11111', features.shape)

        num_vertices = edges_packed.max() + 1

        # print(num_vertices)
        # adj_matrix = torch.zeros((num_vertices, num_vertices))
        connected_matrix = torch.ones((num_vertices, 8)).to(device) * num_vertices
        distance_matrix = torch.zeros((num_vertices, 8)).to(device) * num_vertices
        distance_matrix_ = torch.zeros((num_vertices, 8)).to(device) * num_vertices
        # zeros_m = torch.zeros((1, verts_packed.size(1)))

        # add_verts_packed = torch.cat((verts_packed, zeros_m), dim=0)

        adj_matrix = defaultdict(list)

        for edge in edges_packed:
            #     print(edge)
            #     print(edge[1])
            v1, v2 = edge[0].item(), edge[1].item()
            if v1 not in adj_matrix:
                adj_matrix[v1] = []
                adj_matrix[v1].append(v2)
            else:
                adj_matrix[v1].append(v2)

            if v2 not in adj_matrix:
                adj_matrix[v2] = []
                adj_matrix[v2].append(v1)
            else:
                adj_matrix[v2].append(v1)

        for i in range(num_vertices):
            connected_matrix[i, 0] = torch.tensor(i).long()
            length = len(adj_matrix[i])
            index = torch.tensor(adj_matrix[i]).long()
            connected_matrix[i, 1:length + 1] = index
            # print('index', index.shape)
            # print('1111', verts_packed[index, :])
            # print('2222', verts_packed[index,:][:, 0])
            # print('3333', verts_packed[i,:][0])
            dd = (((verts_packed[i,:][0] - verts_packed[index,:][:, 0]) ** 2 +  (verts_packed[i,:][1] - verts_packed[index,:][:, 1]) ** 2 + (verts_packed[i, :][2] - verts_packed[index, :][:, 2]) ** 2) ** 0.5)
            # print("dddd", dd.shape)
            # dd = dd.transpose(0, 1)
            # print('shape', distance_matrix[i, 1:length + 1].shape)
            distance_matrix[i, 1:length + 1] = dd / dd.sum()
            distance_matrix_[i, 1:length + 1] = dd

        features = self.graph_conv_first(mgh, connected_matrix, distance_matrix).unsqueeze(0)


        return features, connected_matrix, distance_matrix, distance_matrix_



        