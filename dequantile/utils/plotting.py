import numpy as np
import torch
from matplotlib import pyplot as plt
from nflows.utils import tensor2numpy
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import gridspec as grd


def get_bins(data, nbins=20):
    max_ent = data.max().item()
    min_ent = data.min().item()
    return np.linspace(min_ent, max_ent, num=nbins)


def get_bins_two(data, nbins=20, sd=None):
    max_ent = data.max().item()
    min_ent = data.min().item()
    if sd is not None:
        max_ent = max(max_ent, sd.max().item())
        min_ent = min(min_ent, sd.min().item())
    return np.linspace(min_ent, max_ent, num=nbins)


def plot_training(training, validation):
    fig, ax = plt.subplots(1, 1)
    ax.plot(tensor2numpy(training), label='Training')
    ax.plot(tensor2numpy(validation), label='Validation')
    ax.legend()
    return fig


def plot_marginals(originals, sample=None, labels=None, legend=True, axs_nms=None, limits=None, nbins=20):
    data_dim = originals.shape[1]
    n_row = max(data_dim // 4, 1)
    n_col = int(np.ceil(data_dim / n_row))
    fig, axs = plt.subplots(n_row, n_col, figsize=(5 * n_col, 4 * n_row))
    axs = fig.axes
    if labels is None:
        labels = ['original', 'samples']
    for i in range(data_dim):
        if sample is not None:
            bins = get_bins_two(originals[:, i], sd=sample[:, i], nbins=nbins)
        else:
            bins = get_bins_two(originals[:, i], nbins=nbins)
        axs[i].hist(tensor2numpy(originals[:, i]), label=labels[0], alpha=0.5, density=True, bins=bins,
                    histtype='step')
        # Plot samples drawn from the model
        if sample is not None:
            axs[i].hist(tensor2numpy(sample[:, i]), label=labels[1], alpha=0.5, density=True, bins=bins,
                        histtype='step')
        if axs_nms is not None:
            axs[i].set_title(axs_nms[i], fontsize=18)
        if legend:
            axs[i].legend()
        if limits is not None:
            axs[i].set_xlim(limits)
    return fig


def mass_hist(ax, mass, likelihoods, thresholds, cuts, bins=None):
    for t, c in zip(thresholds, cuts):
        mx = likelihoods > t
        survivors = mass[mx.view(-1)]
        if bins is None:
            bins = get_bins(survivors, 50)
        ax.hist(survivors, label=f'{c:.2f}', histtype='step', bins=bins)
    return bins


def mass_hist_normed(ax, mass, likelihoods, thresholds, cuts, bins=None):
    for i, (t, c) in enumerate(zip(thresholds, cuts)):
        mx = likelihoods > t
        survivors = mass[mx.view(-1)]
        if bins is None:
            bins = get_bins(survivors, 50)
            bin_centers = np.convolve(bins, np.ones(2), 'valid')
        # ax.hist(survivors, label=f'{c:.2f}', histtype='step', bins=bins)
        height, bins = np.histogram(survivors, bins=bins)
        if i == 0:
            top = height
        height = height / top
        ax.step(bin_centers, height, label=f'{c:.2f}')
    return bins


def get_weights(data):
    return np.ones_like(data) / len(data)


def mass_hist_cumul(ax, mass, likelihoods, thresholds, cuts, bins=None):
    for i, (t, c) in enumerate(zip(thresholds, cuts)):
        mx = likelihoods > t
        survivors = mass[mx.view(-1)]
        if bins is None:
            bins = get_bins(survivors, 50)
        ax.hist(survivors, label=f'{c:.2f}', histtype='step', bins=bins, weights=get_weights(survivors))
    return bins


def plot_mass_correlation(mass, likelihood, signal_data=None, signal_likelihood=None, cuts=None, normed=0):
    if cuts is None:
        cuts = [1, 0.5, 0.2, 0.1, 0.01]
    cuts = torch.tensor(cuts).cpu()
    # thresholds = likelihood.quantile(cuts)
    thresholds = likelihood.quantile(1 - cuts)
    # thresholds = 1 - cuts
    fig, ax = plt.subplots(1, 1)
    if normed == 1:
        func = mass_hist_normed
    elif normed == 2:
        func = mass_hist_cumul
    else:
        func = mass_hist
    bins = func(ax, mass, likelihood, thresholds, cuts)
    ax.legend()
    return fig


def add_roc_curve(fpr, tpr, ax, label='', auc=None):
    if auc:
        label += f'AUC: {auc:.2f}'
    ax.plot(fpr, tpr, linewidth=2, label=label)


def plot_roc(y_true, y_pred, sv_nm, add_legend=True):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    add_roc_curve(fpr, tpr, ax, auc=auc)

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    if add_legend:
        ax.legend()
    fig.savefig(sv_nm)
    plt.close(fig)

    return auc


def plot_bump(H, Hbkg, Bmin, Bmax, sig, hunter_range, sv_nm):
    # Plot the test histograms with the bump found by BumpHunter plus a little significance plot
    F = plt.figure(figsize=(12, 10))
    gs = grd.GridSpec(2, 1, height_ratios=[4, 1])

    pl1 = plt.subplot(gs[0])

    plt.hist(
        H[1][:-1],
        bins=H[1],
        histtype="step",
        range=hunter_range,
        weights=Hbkg,
        label="background",
        linewidth=2,
        color="red",
    )
    plt.errorbar(
        0.5 * (H[1][1:] + H[1][:-1]),
        H[0],
        xerr=(H[1][1] - H[1][0]) / 2,
        yerr=np.sqrt(H[0]),
        ls="",
        color="blue",
        label="data",
    )

    plt.vlines(
        [Bmin, Bmax],
        0,
        H[0].max(),
        colors="r",
        linestyles='dashed',
        label="BUMP"
    )
    plt.legend(fontsize="xx-large")
    plt.yscale("log")
    if hunter_range is not None:
        plt.xlim(hunter_range)
    plt.xticks(fontsize="xx-large")
    plt.yticks(fontsize="xx-large")
    plt.tight_layout()

    plt.subplot(gs[1], sharex=pl1)
    plt.hist(H[1][:-1], bins=H[1], range=hunter_range, weights=sig)
    plt.plot(np.full(2, Bmin), np.array([sig.min(), sig.max()]), "r--", linewidth=2)
    plt.plot(np.full(2, Bmax), np.array([sig.min(), sig.max()]), "r--", linewidth=2)
    plt.yticks(
        np.arange(np.round(sig.min()), np.round(sig.max()) + 1, step=1),
        fontsize="xx-large",
    )
    plt.ylabel("significance", size="xx-large")
    plt.xticks(fontsize="xx-large")

    plt.savefig(sv_nm, bbox_inches="tight")
    plt.close(F)
