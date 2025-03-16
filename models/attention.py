import torch
from torch import nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionModel, self).__init__()

        self.query_layer = nn.Linear(input_size, hidden_size)
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        attention_scores = torch.matmul(q, k.t()) / (k.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_matrix = torch.matmul(attention_weights, v)

        output = self.output_layer(attended_matrix)

        return output