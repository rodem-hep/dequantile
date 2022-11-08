# Taken from https://github.com/bayesiains/nflows/blob/daee01ade3a32c0921476c1a5db4c9ae56a494d3/nflows/distributions/uniform.py
from typing import Union

import torch

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class BoxUniform(Distribution):
    def __init__(
            self,
            low: Union[torch.Tensor, float],
            high: Union[torch.Tensor, float]
    ):
        """Multidimensionqal uniform distribution defined on a box.

        Args:
            low (Tensor or float): lower range (inclusive).
            high (Tensor or float): upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
        """
        super().__init__()

        if not torch.is_tensor(low):
            low = torch.tensor(low, dtype=torch.float32)
        if not torch.is_tensor(high):
            high = torch.tensor(high, dtype=torch.float32)

        if low.shape != high.shape:
            raise ValueError(
                "low and high are not of the same size"
            )

        if not (low < high).byte().all():
            raise ValueError(
                "low has elements that are higher than high"
            )

        self._shape = low.shape
        self._low = low
        self._high = high
        self._log_prob_value = -torch.sum(torch.log(high - low))

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        return self._log_prob_value.expand(inputs.shape[0])

    def _sample(self, num_samples, context):
        context_size = 1 if context is None else context.shape[0]
        low_expanded = self._low.expand(context_size * num_samples, *self._shape)
        high_expanded = self._high.expand(context_size * num_samples, *self._shape)
        samples = low_expanded + torch.rand(context_size * num_samples, *self._shape) * (high_expanded - low_expanded)

        if context is None:
            return samples
        else:
            return torchutils.split_leading_dim(samples, [context_size, num_samples])
