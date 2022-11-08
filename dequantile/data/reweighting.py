import numpy as np
import torch
from nflows.utils import tensor2numpy



class SingleClassPlaner:

    def __init__(self, variable_index, n_bins=50):
        self.n_bins = n_bins
        self.variable_index = variable_index
        self._is_fit = False
        self.max_pt = None
        self.min_pt = None
        self.bins = None
        self.weights = None

    def get_feature(self, array):
        return array[:, self.variable_index]

    def fit(self, array):
        feature = self.get_feature(array)
        self.max_pt = feature.max().item()
        self.min_pt = feature.min().item()

        step = (self.max_pt - self.min_pt) / self.n_bins
        self.bins = [self.min_pt + step * i for i in range(self.n_bins)]
        self.bins += [self.max_pt + 0.1]
        self.weights, _ = np.histogram(feature, bins=self.bins)
        with np.errstate(divide='ignore'):
            self.weights = np.where(self.weights == 0, 0, len(feature) / (self.weights * self.n_bins))

    def get_weights(self, array):
        # This is applied on a per sample basis
        self.fit(array)
        feature = self.get_feature(array)
        bin_index = np.digitize(tensor2numpy(feature), self.bins) - 1
        return torch.tensor(self.weights[bin_index]).to(array)


class TwoClassPlaner(SingleClassPlaner):

    def get_weights(self, array, labels):
        mx = labels == 0
        weights = torch.zeros(len(array)).to(array)
        weights[mx] = super(TwoClassPlaner, self).get_weights(array[mx])
        weights[~mx] = super(TwoClassPlaner, self).get_weights(array[~mx])
        return weights