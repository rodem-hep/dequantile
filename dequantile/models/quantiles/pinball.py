import torch
from torch import nn


class QuantileRegressor(nn.Module):

    def __init__(self, classifier, network, primary_loss=None, quantiles=None, **kwargs):
        super(QuantileRegressor, self).__init__(**kwargs)
        self.classifier = classifier
        self.network = network
        self.loss_obj = nn.L1Loss(reduction='mean') if primary_loss is None else primary_loss
        if quantiles is None:
            quantiles = [0.5]
        self.register_buffer('quantiles', torch.tensor([0] + quantiles))
        self.cuts = list(range(len(self.quantiles)))

    def forward(self, x, mass):
        """Return the discriminant that the model will use."""
        return (self.network(mass) < x).sum(-1, keepdim=True)

    def transform_to_noise(self, x, mass):
        return self(x, mass), 0

    def compute_loss(self, data, labels, mass, weights, device):
        mass = mass.to(device)
        data = data.to(device)
        x = self.classifier(data, mass)
        predictions = self.network(mass.view(-1, 1))
        mx = labels == 0

        taus = self.quantiles[1:]
        diff = x[mx] - predictions[mx]
        pinball = (taus * torch.max(diff, torch.zeros_like(diff))
                   + (1 - taus) * torch.max(-diff, torch.zeros_like(diff)))

        loss = pinball.sum(-1).mean()
        return loss
