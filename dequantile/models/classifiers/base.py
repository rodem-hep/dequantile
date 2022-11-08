from abc import abstractmethod, ABC

import torch
from torch import nn
import torch.nn.functional as F


class Identity(nn.Module):

    def forward(self, x, *args, **kwargs):
        return x


class Classifier(nn.Module):

    def __init__(self, network, primary_loss=None, encoder=None, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        if encoder is None:
            encoder = Identity()
        self.encoder = encoder
        self.network = network
        self.bce = F.binary_cross_entropy if primary_loss is None else primary_loss

    def forward(self, x, mass):
        """Return the discriminant that the model will use."""
        x = self.encoder(x, mass)
        return self.network(x)

    def regularize(self, data, labels, mass, predictions, weights, loss, device):
        """Overwritten in inheritance to introduce additional loss terms for decorrelation."""
        return loss

    def compute_loss(self, data, labels, mass, weights, device):
        data = data.to(device)
        labels = labels.to(device)
        mass = mass.to(device)
        weights = weights.to(device)
        data = self.encoder(data, mass)
        predictions = self.network(data)
        loss = self.bce(predictions.view(-1), labels, weight=weights).mean()
        loss = self.regularize(data, labels, mass, predictions, weights, loss, device)
        return loss


class Unsupervised(nn.Module, ABC):

    def __init__(self, *args, tail_bound=None, **kwargs):
        """
        A base class for unsupervised methods.
        :param tail_bound: The tail bound of the flow being used for decorrelation.
        """
        super(Unsupervised, self).__init__()
        if isinstance(tail_bound, float) or isinstance(tail_bound, int):
            tail_bound = [-tail_bound, tail_bound]
        self.register_buffer('max_set', torch.zeros(1))
        self.register_buffer('_max', torch.tensor(0, dtype=torch.float32))
        self.register_buffer('_min', torch.tensor(0, dtype=torch.float32))
        self.register_buffer('tail_bound', torch.tensor(tail_bound, dtype=torch.float32))
        self.can_be_sampled = False

    def _set_discrim_scaling(self, lp):
        max_l = lp.max()
        min_l = lp.min()
        if not self.max_set.item():
            self.max_set = torch.ones(1)
            self._max = max_l.detach() if max_l > self._max else self._max
            self._min = min_l.detach() if min_l < self._min else self._min
        else:
            self._max = max_l.detach()
            self._min = min_l.detach()

    @abstractmethod
    def _compute_discriminant(self, data, mass):
        return 0

    def forward(self, x, mass):
        """Return the likelihood scaled by the values seen during training. This is handled in the training."""
        discriminant = self._compute_discriminant(x, mass)
        # Scale to be in range [0, 1]
        scaled_l = (discriminant - self._min) / (self._max - self._min)
        # Scale to be on range [-tail_bound + 0.2, tail_bound - 0.2]
        tb = self.tail_bound + torch.tensor([0.2, -0.2]).to(self.tail_bound)
        return scaled_l.view(-1, 1) * (tb[1] - tb[0]) + tb[0]

    def set_scaling(self, train_loader, device):
        """After a model has been trained, set how to scale the params"""
        for data, label, mass, _ in train_loader:
            data = data.to(device)
            mass = mass.to(device)
            loss = self._compute_discriminant(data, mass)
            self._set_discrim_scaling(loss)

    def compute_loss(self, data, labels, mass, weights, device):
        data = data.to(device)
        mass = mass.to(device)
        loss = self._compute_discriminant(data, mass)
        return loss.mean()
