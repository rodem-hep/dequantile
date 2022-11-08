# !wget https://zenodo.org/record/3606767/files/W_high_level.npz
from collections import defaultdict

import numpy as np

from dequantile.data.base import BaseData
from dequantile.utils.io import on_cluster
from dequantile.utils.torch_utils import shuffle_tensors


class BoostedW(BaseData):

    def split_array(self, array):
        """Split a numpy array into data and labels"""
        array = shuffle_tensors(array)[0]
        data = array[:, :-1]
        # Scale the mass back to its original range
        data[:, 0] = data[:, 0] * 250 + 50
        data[:, -2] = data[:, -2] / 100
        # The labels here are inverted with respect to what is standard, now bkg == 0
        labels = 1 - array[:, -1:]
        if (not on_cluster()) and self.downsample:
            print('Downsampling the data.')
            n_max = int(1e4)
            data = data[:n_max]
            labels = labels[:n_max]
        return [data, labels]

    def _load(self):
        """
        Load the boosted W dataset
        :param unblind: Set this to true once you are ready to put results in the paper
        :return: Numpy files: train, val, test
        """
        data = np.load('dequantile/downloads/W_high_level.npz')
        self._data = defaultdict(list)
        for tag in self._sets:
            self._data[tag] = self.split_array(data[tag].astype('float32'))


class UnsupervisedBoostedW(BoostedW):

    def __init__(self, *args, n_dope=0, scale=None, **kwargs):
        self.n_dope = n_dope
        super(UnsupervisedBoostedW, self).__init__(*args, **kwargs)
        # If this is passed the data will be scaled from [0, 1] to [-self.scale, self.scale]
        self.scale = scale

    def _load(self):
        super(UnsupervisedBoostedW, self)._load()
        train_data, labels = self._data['train']
        bkg_mx = (labels == 0).reshape(-1)
        # Randomly flip n_dope labels and blind yourself to this
        flip_inds = np.random.choice(np.where(~bkg_mx)[0], self.n_dope)
        bkg_mx[flip_inds] = True
        # Set the training set to bkg only + hidden signal
        self._data['train'] = [train_data[bkg_mx], labels[bkg_mx]]

    def get_data_for_loader(self, tag, bkg_only=False):
        data, labels, mass, weights = super(UnsupervisedBoostedW, self).get_data_for_loader(tag, bkg_only)

        return data, labels, mass, weights

