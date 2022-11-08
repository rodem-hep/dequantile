import torch
from nflows import transforms
from nflows.transforms import Transform, CompositeTransform
from torch import nn
from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform as nsf


class ClassifierInn(CompositeTransform):

    def __init__(self, N, num_bins, n_stack=1):
        super().__init__(
            [
                nsf(1, N,
                    num_blocks=2,
                    tail_bound=None,
                    num_bins=num_bins,
                    tails=None,
                    context_features=1)
                for _ in range(n_stack)
            ]
        )


class UnconstrainedClassifierInn(CompositeTransform):

    def __init__(self, N, num_bins, n_stack=1, tail_bound=3.5):
        super().__init__(
            [
                nsf(1, N,
                    num_blocks=2,
                    tail_bound=tail_bound,
                    num_bins=num_bins,
                    tails='linear',
                    context_features=1)
                for _ in range(n_stack)
            ]
        )


class InnEnsemble(Transform):

    def __init__(self, constructor, n_ensemble):
        super(InnEnsemble, self).__init__()
        self.transformers = nn.ModuleList([
            constructor() for _ in range(n_ensemble)
        ])
        self.register_buffer('n_ensemble', torch.tensor(n_ensemble, dtype=torch.float32))

    def _transform(self, transforms, inputs, context):
        z, l = 0, 0
        for func in transforms:
            zt, lt = func(inputs, context)
            z += zt
            l += lt
        return z / self.n_ensemble, l

    def forward(self, inputs, context=None):
        return self._transform(self.transformers, inputs, context)

    def inverse(self, inputs, context=None):
        inverses = (func.inverse for func in self.transformers)
        return self._transform(inverses, inputs, context)


def spline_inn(inp_dim, nodes=128, num_blocks=2, nstack=3, tail_bound=None, tails=None, activation=nn.ReLU(), lu=0,
               num_bins=10, context_features=None):
    transform_list = []
    for i in range(nstack):
        transform_list += [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, nodes, num_blocks=num_blocks,
                                                                               tail_bound=tail_bound, num_bins=num_bins,
                                                                               tails=tails, activation=activation,
                                                                               context_features=context_features)]

        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    return transforms.CompositeTransform(transform_list[:-1])
