# This file contains a simple one dimensional flow that can be used to get the CDF function
import torch
from torch import nn

from dequantile.models.classifiers.flow import ConditionalRqNSF
from dequantile.models.flows.BoxUniform import BoxUniform
from dequantile.utils.torch_utils import CudaDefault


class FlowDecorrelator(nn.Module):
    """
    A wrapper around a 1D flow that can be used to decorrelate a 1D discriminant from any number of other variables.
    """

    def __init__(self, inn, base_density=None, center=0.):
        super(FlowDecorrelator, self).__init__()
        if base_density is None:
            print('Training a flow with a base distribution bounded by [0, 1].')
            base_density = BoxUniform([0], [1])
            center = 0.5
        self.base_density = base_density
        # This defines the center of the base distribution
        self.center = center
        self.transformer = inn
        # This is the log contribution to the likelihood from taking a uniform dist on [0, 1] as the base distribution.
        self.register_buffer('log_p', torch.Tensor(1).log())
        self.cuda_manager = CudaDefault()

    def encode_context(self, context):
        return context

    def transform_to_noise(self, inputs, context):
        context = self.encode_context(context)
        with self.cuda_manager:
            z, logabsdet = self.transformer(inputs, context)
        if not self.training:
            # Get two point in ascending order and test if the function is monotonically increasing.
            lb, _ = self.transformer(0.1 * torch.ones_like(inputs), context)
            rb, _ = self.transformer(0.8 * torch.ones_like(inputs), context)
            # Equivalent to -(z[lb > rb] - self.center) + self.center
            unordered_mx = lb > rb
            z[unordered_mx] = -z[unordered_mx] + 2 * self.center
        return z, logabsdet

    def log_prob(self, inputs, context):
        z, logabsdet = self.transform_to_noise(inputs, context)
        return logabsdet + self.base_density.log_prob(z).to(logabsdet)

    def sample(self, n_samples, context):
        with torch.no_grad():
            context = self.encode_context(context)
            with self.cuda_manager:
                base_samples = self.base_density.sample(n_samples)
                outputs, _ = self.transformer.inverse(base_samples, context)
            return outputs

    def cdf(self, inputs, context):
        # As the base distribution is uniform, the encoding of the flow is equivalent to the cdf.
        outputs, _ = self.transform_to_noise(inputs, context)
        return outputs

    def compute_loss(self, encodings, labels, mass, weights, device):
        encodings = encodings.to(device)
        log_prob = self.log_prob(encodings, mass.view(-1, 1).to(device))
        return -log_prob[labels == 0].mean()

    def forward(self, data, labels, device):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")


class FlowDecorrelatorClassifier(FlowDecorrelator):
    """
    A wrapper around a 1D flow that can be used to decorrelate a 1D discriminant from any number of other variables.
    """

    def __init__(self, classifier, inn, base_density=None, center=0.):
        super(FlowDecorrelatorClassifier, self).__init__(inn, base_density, center)
        self.classifier = classifier

    def compute_loss(self, data, labels, mass, weights, device):
        encodings = self.classifier(data.to(device), mass.to(device))
        return super(FlowDecorrelatorClassifier, self).compute_loss(encodings, labels, mass, weights, device)

    def forward(self, data, labels, device):
        # encodings = self.classifier(data.to(device))
        # prediction, log_det = self.transform_to_noise(encodings, data[:, 4].view(-1, 1).to(device))
        # return prediction
        raise RuntimeError("Forward method cannot be called for a Distribution object.")


class QuantizedFlowDecorrelatorClassifier(FlowDecorrelatorClassifier):

    def __init__(self, classifier, inn, n_bins=10, **kwargs):
        super(QuantizedFlowDecorrelatorClassifier, self).__init__(classifier, inn, **kwargs)
        self.n_bins = n_bins

    def encode_context(self, context):
        return (context * self.n_bins).floor() / self.n_bins


class ConditionalFeatureDecorrelator(ConditionalRqNSF):

    def forward(self, inputs, context):
        return self.flow._transform(inputs, context=context.view(-1, 1))[0]
