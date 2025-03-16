
from torch import nn
import torch
import math

def init_weights(m):
    if type(m) == nn.Linear:
        # nn.init.normal_(m.weight, mean=0, std=1)
        # stdv = 1. / math.sqrt(m.weight.size(1))
        nn.init.uniform_(m.weight, 0, 1)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

class DNN(nn.Module):

    def __init__(self, infeature, numclass):
        super(DNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(numclass, infeature),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(infeature, infeature),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(infeature, infeature),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(infeature, infeature),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            nn.Linear(infeature, numclass),
        )
        # self.classifier.apply(init_weights)

    def forward(self, x):

        x = self.classifier(x)
        return x






