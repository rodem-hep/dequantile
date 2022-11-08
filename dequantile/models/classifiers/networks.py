# Taken from https://github.com/okitouni/MoDe/blob/master/examples/WtaggingTorch.ipynb
from torch import nn


class MoDeModel(nn.Module):
    def __init__(self, input_size=10, output_size=1, activation="silu"):
        """
         DNN Model inherits from torch.torch.nn.Module. Can be initialized with input_size: Number of features per
         sample.

        This is a class wrapper for a simple DNN model. Creates an instance of torch.torch.nn.Module that has 4 nn.Linear
        layers. Use torchsummary for details.

        Parameters
        ----------
        input_size : int=10
            The number of features to train on.
        """
        super().__init__()
        if activation == "silu":
            activ_func = nn.SiLU
        else:
            activ_func = nn.ReLU
        self.model = nn.Sequential(
            nn.Linear(input_size, 64, bias=False),
            activ_func(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            activ_func(),
            nn.Linear(64, 64),
            activ_func(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
