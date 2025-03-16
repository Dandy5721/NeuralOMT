import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv

list_cir = [1,1,1,1,1,1,1]
class GAT_D(torch.nn.Module):
    def __init__(self, in_features, num_layers, nid = 1, heads = 1):
        super().__init__()
        torch.manual_seed(12345)
        list_cir = [1, 1, 1, 1, 1, 1, 1] * num_layers
        self.conv1 = GATConv(in_features, 128, heads, dropout=0.6)
        self.conv1_sub = torch.nn.ModuleList()
        for _ in range(list_cir[0]):
            # self.conv1_sub.append(GATConv(128 * heads, 128, heads, dropout=0.6)) 
            self.conv1_sub.append(GATConv(128* heads, 128, heads, dropout=0.6)) 

        self.conv2 = GATConv(128 * heads, 64, heads, dropout=0.6)
        self.conv2_sub = torch.nn.ModuleList()
        for _ in range(list_cir[1]):
            # self.conv2_sub.append(GATConv(64 * heads, 64, heads, dropout=0.6)) 
            self.conv2_sub.append(GATConv(64 * heads, 64, heads, dropout=0.6))

        self.conv3 = GATConv(64 * heads, 32, heads, dropout=0.6)
        self.conv3_sub = torch.nn.ModuleList()
        for _ in range(list_cir[2]):
            # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) 
            self.conv3_sub.append(GATConv(32 * heads, 32, heads, dropout=0.6)) 

        self.conv4 = GATConv(32 * heads, 16, heads, dropout=0.6)
        self.conv4_sub = torch.nn.ModuleList()
        for _ in range(list_cir[3]):
            # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) # Cora
            self.conv4_sub.append(GATConv(16 * heads, 16, heads, dropout=0.6)) 

        self.conv5 = GATConv(16 * heads, 8, heads, dropout=0.6)
        self.conv5_sub = torch.nn.ModuleList()
        for _ in range(list_cir[4]):
            # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) # Cora
            self.conv5_sub.append(GATConv(8 * heads, 8, heads, dropout=0.6)) 

        self.conv6 = GATConv(8 * heads, 8, heads, dropout=0.6)
        self.conv6_sub = torch.nn.ModuleList()
        for _ in range(list_cir[5]):
            # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) # Cora
            self.conv6_sub.append(GATConv(8 * heads, 8, heads, dropout=0.6)) 

        self.conv7 = GATConv(8 * heads, 4, heads, dropout=0.6)
        self.conv7_sub = torch.nn.ModuleList()
        for _ in range(list_cir[6]):
            # self.conv2_sub.append(GCNConv(2**9, 2**9, cached=True, normalize=not args.use_gdc)) # Cora
            self.conv7_sub.append(GATConv(4 * heads, 4, heads, dropout=0.6)) 

        # # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv8 = GATConv(4 * heads, nid, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.conv1_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        for conv in self.conv2_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        for conv in self.conv3_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        for conv in self.conv4_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv5(x, edge_index))
        for conv in self.conv5_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.elu(conv(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv6(x, edge_index))
        for conv in self.conv6_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv7(x, edge_index))
        for conv in self.conv7_sub:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(x, edge_index))

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv8(x, edge_index)

        return x