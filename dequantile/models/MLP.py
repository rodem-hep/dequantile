from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, N=128, activation=None):
        super().__init__()
        self.hidden_features = N
        layers = [
            nn.Linear(input_dim, N),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Linear(N, output_dim)
        ]
        if activation is not None:
            layers += [activation]
        self.N = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.N(x)
