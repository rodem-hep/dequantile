import torch

from dequantile.models.classifiers.base import Classifier, Unsupervised


# Taken from https://github.com/gkasieczka/DisCo/blob/master/Disco.py
def distance_corr(var_1, var_2, normedweight, power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation

    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries

    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1), len(var_1))
    yy = var_1.repeat(len(var_1), 1).view(len(var_1), len(var_1))
    amat = (xx - yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2), len(var_2))
    yy = var_2.repeat(len(var_2), 1).view(len(var_2), len(var_2))
    bmat = (xx - yy).abs()

    amatavg = torch.mean(amat * normedweight, dim=1)
    Amat = amat - amatavg.repeat(len(var_1), 1).view(len(var_1), len(var_1)) \
           - amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1), len(var_1)) \
           + torch.mean(amatavg * normedweight)

    bmatavg = torch.mean(bmat * normedweight, dim=1)
    Bmat = bmat - bmatavg.repeat(len(var_2), 1).view(len(var_2), len(var_2)) \
           - bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2), len(var_2)) \
           + torch.mean(bmatavg * normedweight)

    ABavg = torch.mean(Amat * Bmat * normedweight, dim=1)
    AAavg = torch.mean(Amat * Amat * normedweight, dim=1)
    BBavg = torch.mean(Bmat * Bmat * normedweight, dim=1)

    if (power == 1):
        dCorr = (torch.mean(ABavg * normedweight)) / torch.sqrt(
            (torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)))
    elif (power == 2):
        dCorr = (torch.mean(ABavg * normedweight)) ** 2 / (
                torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight))
    else:
        dCorr = ((torch.mean(ABavg * normedweight)) / torch.sqrt(
            (torch.mean(AAavg * normedweight) * torch.mean(BBavg * normedweight)))) ** power

    return dCorr


class DisCoClassifier(Classifier, Unsupervised):
    """This inherits from both classes because the classifier output can be a bit weird and needs to be rescaled."""

    def __init__(self, network, alpha=1, primary_loss=None):
        super(DisCoClassifier, self).__init__(network, primary_loss=primary_loss, tail_bound=[0, 1])
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

    def _compute_discriminant(self, data, mass):
        return self.network(data)

    def forward(self, data, mass):
        # return super(Unsupervised, self).forward(data, mass)
        return Unsupervised.forward(self, data, mass)

    def regularize(self, data, labels, mass, predictions, weights, loss, device):
        """Overwritten in inheritance to introduce additional loss terms for decorrelation."""
        # Decorrelate the background only
        mx = labels == 0
        # Apparently the weights should sum to N
        weights = weights[mx]
        scale = weights.mean()
        weights = weights / weights.sum() * mx.sum()
        mode_loss = self.alpha * distance_corr(predictions[mx].view(-1), mass.view(-1, 1)[mx].view(-1),
                                               weights.view(-1))
        return scale * mode_loss + loss
