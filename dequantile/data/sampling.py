# There are at least two different kinds of class you can make for the mass resampling, one to resample s/b in each
# bin such that the fraction is the same everywhere, and a second to resample such that there are the same number of
# samples in each bin. The two can act together to give a flat s/b ratio and a flat mass profile.

from abc import abstractmethod

import numpy as np


class BaseMassSampler:

    def __init__(self, mass, bins):
        self.mass = mass
        # Define the bins
        if bins is None:
            bins = 10
        if isinstance(bins, int):
            bins = np.histogram(mass, bins)[1]
        self.bins = bins
        self.n_bins = len(bins)

    def get_bin_indices(self, array):
        return np.digitize(array, self.bins)

    def sample_arrays(self, fact, *arrays):
        n_points = len(arrays[0])
        n_sample = fact * n_points
        if n_points <= n_sample:
            replace = True
        else:
            replace = False
        return self.shuffle_arrays(*arrays, replace=replace, n_sample=int(n_sample))

    def shuffle_arrays(self, *arrays, replace=False, n_sample=None):
        n_points = len(arrays[0])
        if n_sample is None:
            n_sample = n_points
        index = np.random.choice(np.arange(n_points), size=n_sample, replace=replace)
        return [arr[index] for arr in arrays]

    @abstractmethod
    def resample(self, *args):
        return args


class FlatMassSampler(BaseMassSampler):

    def __init__(self, mass, bins=None, max_resample=3, n_resample=None):
        super(FlatMassSampler, self).__init__(mass, bins)
        self.counts = self.get_bin_counts(self.mass)
        if n_resample is None:
            n_resample = np.mean(self.counts)
        self.n_resample = n_resample
        self.max_resample = max_resample

    def get_bin_counts(self, array):
        return np.histogram(array, self.bins)[0]

    def get_bin_ratios(self, mass):
        return self.n_resample / self.get_bin_counts(mass)

    def resample(self, data, mass, label):
        bin_inds = self.get_bin_indices(mass)
        resampled_mass = []
        resampled_labels = []
        r_data = []
        for i in range(self.n_bins):
            bin_mx = bin_inds == i
            ms = mass[bin_mx]
            lb = label[bin_mx]
            dt = data[bin_mx]
            with np.errstate(divide='ignore'):
                factor = self.n_resample / len(ms)
            if np.isfinite(factor):
                dt, ms, lb = self.sample_arrays(factor, dt, ms, lb)
                resampled_mass += [ms]
                resampled_labels += [lb]
                r_data += [dt]
            else:
                resampled_mass += [ms]
                resampled_labels += [lb]
                r_data += [dt]
        return self.shuffle_arrays(*[np.concatenate(arr) for arr in [r_data, resampled_mass, resampled_labels]],
                                   replace=False)


class RatioMassResampler(BaseMassSampler):
    """
    A class for resampling the mass such that there is a constant S/B ratio in each bin.
    Works for binary classification.

    For a given bin let #S, #B be the un resampled signal and background counts in each bin, and #S_r, #B_r the
    resampled counts.
    This class will set and alpha and beta such that
    #S_r = alpha * #S, #B_r = beta * #B

    This works by ensuring that the s/b in each bin is fixed to some ratio. For each bin this means
    #S_r / #B_r = self.ratio
    after resampling. Where self.ratio defaults to the mean ratio in each bin.
    We want to resample the number of signal events in a given bin by self.alpha such that in that bin the signal is
    sampled to ensure the signal to background ratio is half of the difference between the target ratio and the current
    ratio.
    self.alpha * #S / #B = (self.ratio + #S / #B) / 2

    As we want
    (alpha / beta) * (#S /#B) = self.ratio

    This means we have
    beta = (self.ratio + #S / #B) / (2 * self.ratio)
    """

    def __init__(self, mass, labels, bins=None, max_resample=3, ratio=None):
        super(RatioMassResampler, self).__init__(mass, bins)
        self.labels = labels
        # Assign indicies to which each mass and label belongs
        self._inds = self.get_bin_indices(self.mass)
        # Set the maximum number of times you can resample a given class
        # TODO not using this at all..
        self.max_resample = max_resample
        # Set the s/b ratio to resample to in each bin
        self.ratio = ratio
        # Set all other parameters
        self.fit()

    def get_sb_counts(self, mass, labels):
        # TODO vectorise
        inds = self.get_bin_indices(mass)
        n_bkg = np.zeros_like(self.bins)
        n_sig = np.zeros_like(self.bins)
        for i in range(self.n_bins):
            n_bkg[i] = np.sum(1 - labels[inds == i])
            n_sig[i] = np.sum(labels[inds == i])
        return n_sig, n_bkg

    def get_sb_ratio(self, mass, labels):
        n_sig, n_bkg = self.get_sb_counts(mass, labels)
        return n_sig / n_bkg

    def fit(self):
        n_sig, n_bkg = self.get_sb_counts(self.mass, self.labels)
        self.sb_ratio = n_sig / n_bkg
        if self.ratio is None:
            # The ratio to fix defaults to the mean of the two classes
            self.ratio = np.mean(self.sb_ratio[np.isfinite(self.sb_ratio)])
            print(f'Ratio set to {self.ratio}.')
        self.alpha = np.abs((self.ratio + self.sb_ratio) / (2 * self.sb_ratio))
        self.beta = np.abs((self.ratio + self.sb_ratio) / (2 * self.ratio))

    def resample(self, data, mass, labels):
        # TODO also need to resample data/features...
        # TODO need to tidy house here
        bin_inds = self.get_bin_indices(mass)
        resampled_mass = []
        resampled_labels = []
        r_data = []
        for i in range(self.n_bins):
            bin_mx = bin_inds == i
            ms = mass[bin_mx]
            lb = labels[bin_mx]
            dt = data[bin_mx]
            for fact, idf in zip([self.alpha[i], self.beta[i]], [1, 0]):
                mx = lb == idf
                if np.isfinite(fact):
                    r_d, rs_ms, rs_lb = self.sample_arrays(fact, dt[mx], ms[mx], lb[mx])
                    resampled_mass += [rs_ms]
                    resampled_labels += [rs_lb]
                    r_data += [r_d]
                else:
                    resampled_mass += [ms[mx]]
                    resampled_labels += [lb[mx]]
                    r_data += [dt[mx]]
        return self.shuffle_arrays(*(np.concatenate(arr) for arr in [r_data, resampled_mass, resampled_labels]),
                                   replace=False)


class FlatRatioSampler(BaseMassSampler):

    """
    This class combines the other two classes in this file. This kind of sampling isn't necessarily the best way to
    approach the problem as the number of times that each samples has been sampled in the previous sampler isn't taken
    into account. In practise it appears to perform fine as is.
    """

    def __init__(self, mass, labels, bins=None, ratio=None):
        super(FlatRatioSampler, self).__init__(mass, bins)
        self.ratio_resample = RatioMassResampler(mass, labels, bins=bins, ratio=ratio)
        self.flat_resample = FlatMassSampler(mass, bins=bins)

    def resample(self, data, mass, labels):
        data, mass, labels = self.ratio_resample.resample(data, mass, labels)
        return self.flat_resample.resample(data, mass, labels)
