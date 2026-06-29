import torch
from torch import nn




class MpcBcNetwork(nn.Module):
    def __init__(self, n_states, n_controls):
        super(MpcBcNetwork,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_states, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, n_controls),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.model(x)
        return x


