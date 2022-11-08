import numpy as np
import torch
from matplotlib import pyplot as plt
from nflows.utils import tensor2numpy
from scipy.stats import entropy

from dequantile.utils.torch_utils import shuffle_tensors
import pyBumpHunter as pbh


class Metrics():
    def __init__(self, validation=False):
        self.validation = validation
        self.losses = []
        self.accs = []
        self.signalE = []
        self.backgroundE = []
        if self.validation:
            self.R50 = []
            self.JSD = []

    def calculate(self, pred, target, l=None, m=None):
        preds = np.array(pred.tolist()).flatten()
        targets = np.array(target.tolist()).flatten()
        acc = (preds.round() == targets).sum() / targets.shape[0]
        signal_efficiency = ((preds.round() == targets) & (targets == 0)).sum() / (targets == 1).sum()
        background_efficiency = ((preds.round() == targets) & (targets == 1)).sum() / (targets == 0).sum()
        if self.validation:
            c = find_threshold(preds, (targets == 0), 0.5)
            R50 = 1 / ((preds[targets == 1] < c).sum() / (targets == 1).sum())
            self.R50.append(R50)
            if m is not None:
                m = np.array(m.tolist()).flatten()[targets == 1]
                bkg_preds = preds[targets == 1]
                hist1, bins = np.histogram(m[bkg_preds > c], bins=50, density=True)
                hist2, _ = np.histogram(m[bkg_preds < c], bins=bins, density=True)
                # Mask out the bad
                mx = (hist1 > 0) & (hist2 > 0)
                hist1 = hist1[mx]
                hist2 = hist2[mx]
                JSD = 0.5 * (entropy(hist1, 0.5 * (hist1 + hist2)) + entropy(hist2, 0.5 * (hist1 + hist2)))
                self.JSD.append(JSD)

                n1 = np.sum((targets == 1) & (preds > c))
                n_ones = np.sum(targets == 1)
                JSDs = []
                for i in range(10):
                    index = np.random.permutation(n_ones)
                    hist1, bins = np.histogram(m[index[:n1]], bins=50, density=True)
                    hist2, _ = np.histogram(m[index[n1:]], bins=bins, density=True)
                    JSDs += [0.5 * (entropy(hist1, 0.5 * (hist1 + hist2)) + entropy(hist2, 0.5 * (hist1 + hist2)))]
        self.accs.append(acc)
        self.signalE.append(signal_efficiency)
        self.backgroundE.append(background_efficiency)
        if l:
            self.losses.append(l)
        return JSDs


def metric_calc(model, X, y, x_biased, metric=None):
    predictions = np.array(model(X).tolist()).flatten()
    if metric is None:
        metrics_test = Metrics(validation=True)
    else:
        metrics_test = metric(validation=True)
    metrics_test.calculate(pred=predictions, target=y, m=x_biased)
    R50 = metrics_test.R50[0]
    JSD = metrics_test.JSD[0]
    return (1 / JSD, R50)


def find_threshold(L, mask, x_frac):
    """
    Calculate c such that x_frac of the array is less than c.

    Parameters
    ----------
    L : Array
        The array where the cutoff is to be found
    mask : Array,
        Mask that returns L[mask] the part of the original array over which it is desired to calculate the threshold.
    x_frac : float
        Of the area that is lass than or equal to c.

    returns c (type=L.dtype)
    """
    max_x = mask.sum()
    x = int(np.round(x_frac * max_x))
    L_sorted = np.sort(L[mask.astype(bool)])
    return L_sorted[x]


def plot_mode_metrics(predictions, label, x_biased, file_name, mbins=100, histbins=None, name='', legend_ncol=3,
                      ptype=0, dynamicbins=True, scale=None, xlim=None):
    predictions, label = tensor2numpy(predictions.view(-1)), tensor2numpy(label.view(-1))
    # This assumes that the labels are switched back
    label = 1 - label
    predictions = 1 - predictions

    if histbins == None:
        histbins = mbins
    predictions = np.array(predictions).flatten()
    metrics_test = Metrics(validation=True)
    JSDs = metrics_test.calculate(pred=predictions, target=label, m=x_biased)
    R50 = metrics_test.R50[0]
    JSD = metrics_test.JSD[0]

    c = find_threshold(predictions, (label == 0), 0.5)
    fig, [ax, ax1] = plt.subplots(2, 1, figsize=(6, 8), dpi=120, sharex=True)
    _, bins, _ = ax.hist(x_biased[(label == 1)], bins=histbins, alpha=0.3, color='C1', label='Background', density=True,
                         log=True)
    ax.hist(x_biased[(label == 1) & (predictions < c)], bins=bins, alpha=0.3, color='C0', label='False Positives',
            density=True, log=True)
    ax.set_ylabel("Normed Counts", fontsize=14)
    ax.set_title('{} (R50:{:.2f}, 1/JSD:{:.2f})'.format(name, R50, 1 / JSD))
    ax.legend()

    efficiencies = np.linspace(0.1, 0.9, 9)[::-1]
    cuts = []
    for eff in efficiencies:
        cuts.append(find_threshold(predictions, (label == 0), eff))
    m = x_biased[label == 1]
    scores = predictions[label == 1]
    if dynamicbins:
        mod = len(m) % mbins
        if mod != 0:
            m = m[:-mod]
            scores = scores[:-mod]
        sorted_m = np.argsort(m)
        scores = scores[sorted_m].reshape(mbins, -1)
        m = m[sorted_m]
        m = m.reshape(mbins, -1).mean(axis=1)
    else:
        _, bins = np.histogram(m, bins=mbins)
        digitized = np.digitize(m, bins)
        m = (bins[:-1] + bins[1:]) * 0.5
    for j, cut in enumerate(cuts):
        c = f"C{j}"  # if j!= 6 else f"C11"
        if dynamicbins:
            passed = (scores < cut).sum(axis=1) / scores.shape[1]
        else:
            passed = [(scores[digitized == i + 1] < cut).sum() / (digitized == i + 1).sum() for i in range(mbins)]
        if ptype == 0:
            ax1.plot(m, passed, label="{:0.1f}".format(efficiencies[j]), alpha=0.9, c=c, lw=1)
        else:
            ax1.step(m, passed, label="{:0.1f}".format(efficiencies[j]), alpha=0.9, c=c, lw=1)
    if scale is not None:
        ax1.set_yscale(scale)
    ax1.set_ylabel("False Pos. Rate", fontsize=14)
    ax1.set_ylim([-0.02, 1.02])
    if xlim: ax1.set_xlim(xlim)
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), markerscale=5, title="True Pos. Rate (Signal Efficiency)",
               ncol=legend_ncol)
    ax1.set_xlabel(r"$m$", fontsize=14)
    fig.tight_layout(pad=0)
    fig.savefig(file_name)
    return (1 / JSD, R50, JSDs)


def run_bump_hunts(predictions, labels, mass, plot_dir):
    predictions = predictions.cpu()
    labels = labels.cpu()
    # Shuffle the predictions and masses
    mass, predictions, labels = shuffle_tensors(mass, predictions, labels)

    # Split into signal and background
    def mx_data(label, *args):
        return [t[labels == label] for t in args]

    bkg_masses, bkg_pred = mx_data(0, mass, predictions)
    sig_masses, sig_pred = mx_data(1, mass, predictions)
    # Define thresholds based on the background predictions
    # TODO switch back to this
    quantiles = [0.0, 0.5, 0.9, 0.95, 0.99]
    # quantiles = [0.99]
    cuts = bkg_pred.quantile(torch.Tensor(quantiles).to(bkg_pred))
    # bkg_masses[:n_bkg], bkg_pred[:n_bkg]

    # Define an object with which we can do
    signal_strengths = []
    significances = []
    sig_errors = []
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    bump_dir = plot_dir / 'bumps'
    bump_dir.mkdir(exist_ok=True)
    for cut, q in zip(cuts, quantiles):
        sig_mass = sig_masses[sig_pred >= cut]
        p_bkg = bkg_masses[bkg_pred >= cut]
        hunter = pbh.BumpHunter1D(
            rang=[m.item() for m in [mass.min(), mass.max()]],
            width_min=2,
            width_max=10,
            width_step=1,
            scan_step=1,
            npe=10000,
            nworker=1,
            seed=666,
            use_sideband=1
        )

        # hunter.bump_scan(m_bkg, p_bkg)

        # TODO how to set these parameters?
        # We have to set additionnal parameters specific to the signal injection.
        # All the parameters defined previously are kept.
        hunter.sigma_limit = 2
        hunter.str_min = 1  # if str_scale='log', the real starting value is 10**str_min
        hunter.str_scale = 'log'
        hunter.signal_exp = 1  # Corresponds to the real number of signal events generated when making the data
        hunter.npe_inject = 2000

        # TODO you don't actually need to split the data into pieces for this, it will sample the bkg as needed
        hunter.signal_inject(sig_mass, p_bkg, is_hist=False)

        # Get the signal strength at each point in the scan
        signal_strengths += [np.arange(
            hunter.str_min,
            hunter.str_min + hunter.str_step * len(hunter.sigma_ar),
            step=hunter.str_step,
        )]
        # Corresponding significance
        significances += [hunter.sigma_ar[:, 0]]
        sig_errors += [hunter.sigma_ar[:, 1:]]

        ax.errorbar(
            signal_strengths[-1],
            significances[-1],
            xerr=0,
            yerr=[sig_errors[-1][:, 0], sig_errors[-1][:, 1]],
            marker='o',
            linewidth=2,
            label=f'{q}'
        )
        # Inject the number of signal events that was required to see the given excess
        p_bkg, m_bkg = np.array_split(p_bkg, 2)
        signal_injected = np.concatenate((m_bkg, sig_mass[:int(hunter.signal_ratio * hunter.signal_exp / 2)]))
        hunter.plot_bump(signal_injected, p_bkg, filename=bump_dir / f'bump_{q}.png')
    ax.set_title("Significance vs signal strength", size="xx-large")
    ax.set_xlabel("Signal strength", size="xx-large")
    ax.set_ylabel("Significance", size="xx-large")
    ax.set_xscale('log')
    ax.legend()

    fig.savefig(plot_dir / 'signal_injection.png')
    plt.close(fig)

    return signal_strengths, significances, sig_errors
