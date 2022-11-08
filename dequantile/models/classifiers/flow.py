import torch
from nflows.distributions import StandardNormal
from nflows.flows import Flow

from dequantile.models.classifiers.base import Unsupervised
from dequantile.models.flows.inns import spline_inn


class RqNsf(Unsupervised):

    def __init__(self, input_dim, decor_tb, nstack=4, tails='linear', tail_bound=3.5, num_bins=8, n_cond=0):
        super(RqNsf, self).__init__(tail_bound=decor_tb)
        self.flow = Flow(
            spline_inn(input_dim, nstack=nstack, tails=tails, tail_bound=tail_bound, num_bins=num_bins,
                       context_features=n_cond),
            StandardNormal([input_dim])
        )
        self.can_be_sampled = True

    def sample(self, num, context=None, batch_size=None):
        return self.flow.sample(num, batch_size=batch_size)

    def log_prob(self, inputs, context=None):
        return self.flow.log_prob(inputs, context=None)

    def _compute_discriminant(self, data, mass):
        lp = self.log_prob(data, mass)
        return -lp


class ConditionalRqNSF(RqNsf):

    def __init__(self, *args, **kwargs):
        super(ConditionalRqNSF, self).__init__(*args, n_cond=1, **kwargs)

    def log_prob(self, inputs, context=None):
        return self.flow.log_prob(inputs, context=context.view(-1, 1))

    def sample(self, num, context=None, batch_size=None):
        return self.flow.sample(num, context.view(-1, 1), batch_size=batch_size)
