import torch

from dequantile.models.classifiers.base import Classifier
from modeloss.pytorch import MoDeLoss

from dequantile.utils.torch_utils import mask_data


class MoDeClassifier(Classifier):

    def __init__(self, network, order, alpha=1, primary_loss=None):
        super(MoDeClassifier, self).__init__(network, primary_loss=primary_loss)
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self.mode_loss = MoDeLoss(order=order, background_only=True)

    def regularize(self, data, labels, mass, predictions, weights, loss, device):
        """Overwritten in inheritance to introduce additional loss terms for decorrelation."""
        mode_loss = self.alpha * self.mode_loss(predictions, labels.view(-1, 1), mass.view(-1, 1), weights.view(-1, 1))
        return mode_loss + loss
