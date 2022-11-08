from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from sklearn.utils import compute_sample_weight
from torch.utils.data import DataLoader

from dequantile.data.reweighting import TwoClassPlaner
from dequantile.data.sampling import RatioMassResampler, FlatRatioSampler, FlatMassSampler
from dequantile.utils.torch_utils import TupleDataset


class BaseData(ABC):

    def __init__(self, preprocessor, unblind=False, drop_mass=False, drop_pt=False, resample=2, downsample=True,
                 use_weights=True, scale=None, center_mass=False):
        super(BaseData, self).__init__()
        # TODO the feature names and mass index aren't generic
        self.center_mass = center_mass
        self.scale = scale
        self.use_weights = use_weights
        self.downsample = downsample
        self.feature_names = ['mass', 'pt', 'tau21', 'c2', 'd2', 'fw', 'pf', 'ap', 'zcutdef', 'ktdr', 'sqrtd12',
                              'label']
        self.mass_index = 0
        self._sets = ['train', 'val', 'test']
        self.unblind = unblind
        self.preprocessor = preprocessor
        self.data_dim = len(self.feature_names) - 1
        self.load()
        self.preprocessor.fit(self._data['train'][0])
        self._drop_mass = drop_mass
        self._drop_pt = drop_pt
        self.__un_data_dim = self.data_dim

        if drop_mass:
            self.data_dim = self.data_dim - 1
        if drop_pt:
            self.data_dim = self.data_dim - 1

        # Set the initial batch sizes to something sensible
        self.train_batch_size = 256
        self.eval_batch_size = 1000
        self.bkg_only = False
        self.update_sampler(resample)

    def update_sampler(self, resample):
        self.resample = resample
        if self.resample > 0:
            dt, lt = self._data[self._sets[0]]
            if self.bkg_only:
                bkg_mx = lt == 0
                dt = dt[bkg_mx.reshape(-1)]
                lt = lt[bkg_mx]
            mt = self.get_mass(dt)
        if self.resample == 1:
            self._resampler = RatioMassResampler(mt, lt, bins=50)
        if self.resample == 2:
            self._resampler = FlatMassSampler(mt, bins=50)
        if self.resample == 3:
            self._resampler = FlatRatioSampler(mt, lt, bins=50)

    def get_mass(self, data):
        return data[:, self.mass_index]

    def remove_mass(self, data):
        return np.delete(data, self.mass_index, 1)

    @abstractmethod
    def _load(self):
        """Load the data set and assign self._data[tag] = [data, label] for all tags"""
        # self._data = defaultdict(list)
        # for tag in self.__sets:
        #     self._data[tag] = [data_tag, label_tag]
        return None

    def load(self):
        """
        :param unblind: Set this to true once you are ready to put results in the paper
        :return: Numpy files: train, val, test
        """
        self._load()
        if not self.unblind:
            self._data['test'] = deepcopy(self._data['val'])

    def get_transformed(self, tag, tensor=True):
        data, label = self._data[tag]
        if self.resample > 0:
            data, _, label = self._resampler.resample(data, self.get_mass(data), label.reshape(-1))
        data = self.preprocessor.transform(data)
        if tensor:
            return torch.Tensor(data), torch.Tensor(label).view(-1)
        else:
            return data, label

    def get_preprocessed_data(self, tensors=True):
        return [self.get_transformed(tag, tensors) for tag in self._sets]

    def unscale_mass(self, masses):
        shell = np.zeros((len(masses), self.__un_data_dim))
        shell[:, self.mass_index] = masses
        return self.get_mass(self.preprocessor.inverse_transform(shell))

    def get_data_for_loader(self, tag, bkg_only=False):
        data, labels = self.get_transformed(tag, tensor=True)
        mass = self.get_mass(data)
        if self.center_mass:
            mass = 2 * mass - 1
        if self.use_weights == 1:
            weights = torch.tensor(compute_sample_weight('balanced', labels)).to(mass).view(-1)
        elif self.use_weights == 2:
            weights = TwoClassPlaner(1).get_weights(data, labels)
        else:
            weights = torch.ones_like(mass).view(-1)

        if self._drop_mass:
            data = self.remove_mass(data)
            if self._drop_pt:
                data = data[:, 1:]
        elif self._drop_pt:
            data = np.delete(data, 1, 1)

        if bkg_only:
            bkg_mx = labels == 0
            data = data[bkg_mx]
            mass = mass[bkg_mx]
            labels = labels[bkg_mx]
            weights = torch.ones_like(labels)
        if self.scale is not None:
            data = data * self.scale * 2 - self.scale
        return data, labels, mass, weights

    def get_loader(self, tag, batch_size, bkg_only=False):
        data, labels, mass, weights = self.get_data_for_loader(tag, bkg_only)
        return DataLoader(dataset=TupleDataset(data, labels, mass, weights),
                          batch_size=batch_size,
                          num_workers=0,
                          shuffle=True,
                          drop_last=True  # for MoDe the last batch causes problems, so drop it for all models
                          )

    def setup_loaders(self, train_batch_size, eval_batch_size, bkg_only=False):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.bkg_only = bkg_only

    def get_loaders(self):
        """Return train loader, valid loader, test loader"""
        return [self.get_loader(tag, bs, self.bkg_only) for tag, bs in
                zip(self._sets, [self.train_batch_size] + [self.eval_batch_size] * 2)]
