import torch
import torch.nn as nn

class FAFWNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.relu(self.fc1(x))
        gates = self.sigmoid(self.fc2(gates))
        x = torch.mul(x, gates)
        return {
            "output": x,
            "attentions": gates,
        }