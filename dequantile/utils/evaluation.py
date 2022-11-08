from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from nflows.utils import tensor2numpy
from scipy.special import gammainc
from scipy.stats import norm, entropy
from sklearn.metrics import roc_auc_score
import pyBumpHunter as pbh

from dequantile.utils.io import on_cluster
from dequantile.utils.metrics import plot_mode_metrics
from dequantile.utils.plotting import plot_marginals, plot_mass_correlation, plot_roc, plot_bump, get_bins
from dequantile.utils.torch_utils import shuffle_tensors


def plot_correlations(predictions, labels, mass, plots_dir):
    # Plot the two likelihood distributions
    fig = plot_marginals(predictions[labels == 0].view(-1, 1), predictions[labels == 1].view(-1, 1),
                         labels=['QCD', 'Signal'])
    fig.savefig(plots_dir / f'likelihoods.png')
    plt.close(fig)

    # Look at some mass correlation plots
    nominal_preds = predictions[labels == 0].detach().cpu()
    nominal_masses = mass[labels == 0]
    fig = plot_mass_correlation(nominal_masses, nominal_preds)
    fig.savefig(plots_dir / 'mass_correlation.png')
    plt.close(fig)

    fig = plot_mass_correlation(nominal_masses, nominal_preds, normed=1)
    fig.savefig(plots_dir / 'mass_correlation_normed.png')
    plt.close(fig)

    fig = plot_mass_correlation(nominal_masses, nominal_preds, normed=2)
    fig.savefig(plots_dir / 'mass_correlation_cumul.png')
    plt.close(fig)

    # ROC plots and save preds
    preds = tensor2numpy(predictions)
    truth = tensor2numpy(labels)
    np.save(plots_dir / 'pred.npy', preds)
    np.save(plots_dir / 'truth.npy', truth)
    np.save(plots_dir / 'mass.npy', mass)
    plot_roc(truth, preds, plots_dir / 'auc.png')

    # ROC plots in mass bins
    n_bins = 20
    _, bins = np.histogram(mass, bins=n_bins)
    aucs = []
    masses = []
    p_signal = []
    for i in range(n_bins):
        mx = ((mass < bins[i + 1]) & (mass >= bins[i])).reshape(-1, 1)
        try:
            bin_label = truth[mx]
            aucs += [roc_auc_score(bin_label, preds[mx])]
            masses += [(bins[i + 1] + bins[i]) / 2]
            p_signal += [np.sum(bin_label)]

        except Exception as e:
            # Sometimes there is no signal in a given mass bin and so we will ignore that bin
            print(e)
    # Get the mass weighted AUC scores
    total_signal = np.sum(truth)
    mw_aucs = np.array(aucs) * np.array(p_signal) / total_signal
    mw_auc = np.sum(mw_aucs)
    np.save(plots_dir / 'mw_auc.npy', mw_auc)

    fig, ax = plt.subplots()
    ax.plot(masses, aucs, label=f'AUC: {mw_auc}')
    ax.legend()
    fig.savefig(plots_dir / 'auc_mass.png')
    plt.close(fig)


# Run a bump hunt using the classifiers we have just defined.
def run_bump_hunt(predictions, labels, mass, plot_dir, random_cuts=True, use_sideband=False, init_sig=2, cuts=None,
                  quantiles=None, sub_dir='bumps'):
    predictions = predictions.cpu()
    labels = labels.cpu()
    # Shuffle the predictions and masses
    mass, predictions, labels = shuffle_tensors(mass, predictions, labels)

    # Split into signal and background
    def mx_data(label, *args):
        return pd.DataFrame.from_dict(dict(zip(['mass', 'predictions'], [t[labels == label] for t in args])))

    # TODO you could make this inside the inner hunter loop, that way you also have different signal events each hunt
    bkg = mx_data(0, mass, predictions)
    sig = mx_data(1, mass, predictions)
    n_bkg = bkg.shape[0]
    if not random_cuts:
        n_bkg = n_bkg // 2
    n_sig = int(init_sig * n_bkg ** 0.5)
    print(f'Added {n_sig} signal events to {n_bkg} bkg events.')
    data = pd.concat((bkg[:n_bkg], sig[:n_sig]))
    if not random_cuts:
        bkg = bkg[n_bkg:]
    # Define thresholds based on the background predictions
    if quantiles is None:
        quantiles = [0.0, 0.5, 0.9, 0.95, 0.99]
    if cuts is None:
        cuts = bkg['predictions'].quantile(quantiles)
    # Define directory for saving
    bump_dir = plot_dir / sub_dir
    bump_dir.mkdir(exist_ok=True, parents=True)
    # TODO kwarg
    if random_cuts:
        # This will give multiple seeds for multiple samples of the bkg
        n_hunts = 5
    else:
        n_hunts = 1
    # Run a bump hunt for each cut value
    for cut, q in zip(cuts, quantiles):
        info_dict = defaultdict(list)
        for hunt in range(n_hunts):
            # Get the masses that pass the cut
            data_mass = data['mass'][data['predictions'] >= cut]
            if random_cuts:
                p_bkg = data['mass'].sample(frac=1 - q)
            else:
                p_bkg = bkg['mass'][bkg['predictions'] >= cut]

            # Define a bump hunting object
            hunter = pbh.BumpHunter1D(
                rang=[m.item() for m in [mass.min(), mass.max()]],
                width_min=2,
                width_max=10,
                width_step=1,
                scan_step=1,
                npe=10000,
                nworker=1,
                use_sideband=use_sideband,
                seed=42 + hunt
            )

            hunter.bump_scan(data_mass, p_bkg)
            # hunter.plot_bump(data_mass, p_bkg, filename=bump_dir / f'bump_hunt_{hunt}_{q}.png')
            # Pulling out the pbh plotting and remaking it
            H = np.histogram(data_mass, bins=hunter.bins, range=hunter.rang)
            Bmin = H[1][hunter.min_loc_ar[0]]
            Bmax = H[1][hunter.min_loc_ar[0] + hunter.min_width_ar[0]]
            Hbkg = np.histogram(p_bkg, bins=hunter.bins, range=hunter.rang, weights=hunter.weights)[0]
            info_dict['global_significance'] += [hunter.significance]
            info_dict['bump_size'] += [hunter.signal_eval]
            info_dict['bump_center'] += [(Bmax + Bmin) / 2]

            if use_sideband:
                Hbkg = Hbkg * hunter.norm_scale

            # Calculate significance for each bin
            sig = np.ones(Hbkg.size)
            sig[(H[0] > Hbkg) & (Hbkg > 0)] = gammainc(
                H[0][(H[0] > Hbkg) & (Hbkg > 0)], Hbkg[(H[0] > Hbkg) & (Hbkg > 0)]
            )
            sig[H[0] < Hbkg] = 1 - gammainc(H[0][H[0] < Hbkg] + 1, Hbkg[H[0] < Hbkg])
            sig = norm.ppf(1 - sig)
            sig[sig < 0.0] = 0.0  # If negative, set it to 0
            np.nan_to_num(sig, posinf=0, neginf=0, nan=0, copy=False)  # Avoid errors
            sig[H[0] < Hbkg] = -sig[H[0] < Hbkg]  # Now we can make it signed

            # Stash the results
            keys = ['H', 'Hbkg', 'Bmin', 'Bmax', 'sig']
            for key, value in zip(keys[1:], [Hbkg, Bmin, Bmax, sig]):
                info_dict[key] += [value]
            info_dict['H0'] += [H[0]]
            info_dict['H1'] += [H[1]]
            info_dict['hunter_range'] += [hunter.rang]

        # Plot the test histograms with the bump found by BumpHunter plus a little significance plot
        # TODO can a bump hunt be straightforwardly ensembled like this?
        np.save(bump_dir / f'hunt_dict_{int(q * 100)}.npy', info_dict)
        new_dict = {}
        for key in keys[1:] + ['H0', 'H1', 'hunter_range']:
            new_dict[key] = np.mean(np.vstack(info_dict[key]), 0)
        for key in ['global_significance', 'bump_size', 'bump_center']:
            new_dict[key] = np.mean(info_dict[key])
        # np.save(bump_dir / f'hunt_dict_{int(q * 100)}.npy', new_dict)
        # Load with
        # tt = np.load(bump_dir / f'hunt_dict_{int(q * 100)}.npy', allow_pickle=True).item()
        new_dict['H'] = (new_dict['H0'], new_dict['H1'])
        plot_bump(*[new_dict[key] for key in keys], hunter.rang, bump_dir / f'bump_{q}.png')

def plot_multi_cuts(data, save_dir):
    thresholds = np.unique(data['quantiles'])[1:]
    counts = (np.array(data['quantiles']).reshape(-1, 1) >= thresholds.reshape(1, -1)).sum(0)
    n_data = data.shape[0]

    n_plots = len(thresholds)
    # fig, ax = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    fig, ax = plt.subplots(2, n_plots // 2, figsize=(6 * n_plots, 18))
    ax = fig.axes
    bins = [None] * n_plots
    heights = defaultdict(list)

    # Get the background counts
    for j in range(100):
        for i, count in enumerate(counts):
            data = data.sample(frac=1)
            take = data['mass'][:count]
            if bins[i] is None:
                n_bins = 30
                if i >= len(counts) - 1:
                    n_bins = 20
                bins[i] = get_bins(take, n_bins)
            cnts, _ = np.histogram(take, bins=bins[i])
            heights[i] += [cnts]

    # Make a filled histogram for the background
    minimums = []
    maximums = []
    for i in range(len(counts)):
        hts = np.vstack(heights[i])
        minimums += [hts.min(0)]
        maximums += [hts.max(0)]

    for i in range(len(counts)):
        def recast(bb, tt):
            bns = (bb.reshape(-1, 1) * np.ones(2)).reshape(-1)[1:-1]
            hts = (tt.reshape(-1, 1) * np.ones(2)).reshape(-1)
            return bns, hts

        bns, mn_hts = recast(bins[i], minimums[i])
        bns, max_hts = recast(bins[i], maximums[i])
        ax[i].fill_between(bns, mn_hts, max_hts, color='#CC79A7', alpha=0.3, label='Random')

    # Plot the decorrelators
    label_key = {'encodings': 'DisCo', 'quantiles': 'qr-DisCo', 'predictions': 'cf-DisCo'}
    for decorrelator in ['encodings', 'quantiles', 'predictions']:
        data = data.sort_values(decorrelator)
        for i, count in enumerate(counts):
            take = data['mass'][:count]
            ax[i].hist(take, histtype='step', label=label_key[decorrelator], bins=bins[i], linewidth=1.5)
            ax[i].set_title(f'Background rejection: {1 - count / n_data:.3f}%')
            ax[i].set_yscale('log')
            ax[i].set_xlabel('Mass [GeV]')

    ax[1].legend()
    ax[0].set_ylabel('Normalized counts')
    ax[2].set_ylabel('Normalized counts')
    fig.tight_layout()
    fig.savefig(save_dir / 'multi_cuts.pdf')


def run_evaluation(classifier, flow, quantiler, test_loader, device, mass_unscaler, results_directory,
                   use_sideband=True):
    # Get the likelihoods of the outliers and test set
    enc = []
    predictions = []
    labels = []
    mass = []
    quantiles = []
    with torch.no_grad():
        for t_step, (t_data, t_label, t_mass, _) in enumerate(test_loader):
            encodings = classifier(t_data.to(device), t_mass.to(device))
            enc += [encodings]
            predictions += [flow.transform_to_noise(encodings, t_mass.view(-1, 1).to(device))[0]]
            labels += [t_label]
            mass += [t_mass]
            quantiles += [quantiler(encodings, t_mass.view(-1, 1).to(device))]
    encodings = torch.cat(enc, dim=0).view(-1, 1)
    predictions = torch.cat(predictions, dim=0).view(-1, 1)
    labels = torch.cat(labels, dim=0).view(-1, 1)
    quantiles = torch.cat(quantiles, dim=0).view(-1, 1).to(encodings)
    mass = mass_unscaler(torch.cat(mass, dim=0).cpu().numpy()).reshape(-1, 1)

    # Dump everything to hdf as a dataframe
    data = pd.DataFrame({
        'mass': mass.reshape(-1),
        'encodings': tensor2numpy(encodings).reshape(-1),
        'labels': tensor2numpy(labels).reshape(-1),
        'predictions': tensor2numpy(predictions).reshape(-1),
        'quantiles': tensor2numpy(quantiles).reshape(-1),
    })
    data.to_hdf(results_directory / 'results_df.h5', key='df', index=False)

    # TODO from here there is quite a bit of code duplication, this is different functions on the same data
    quantile_dir = results_directory / 'quantiles'
    quantile_dir.mkdir(exist_ok=True)
    plot_correlations(predictions, labels, mass, quantile_dir)
    encodings_dir = results_directory / 'encodings'
    encodings_dir.mkdir(exist_ok=True)
    plot_correlations(encodings, labels, mass, encodings_dir)

    # Plot some of the metrics from mode
    corr_dir = results_directory / 'mode_metrics'
    corr_dir.mkdir(exist_ok=True)
    d_jsd, dr50, dpJSDs = plot_mode_metrics(predictions, labels, mass, corr_dir / 'quantiles.png')
    c_jsd, cr50, cpJSDs = plot_mode_metrics(encodings, labels, mass, corr_dir / 'encodings.png')
    np.save(corr_dir / 'mode_metrics.npy', np.array([d_jsd, dr50, c_jsd, cr50]))


    ####################################################################################################################
    # Calculate R_50, 1/JSD metrics by hand
    encodings = encodings.cpu().numpy()
    labels = labels.cpu().numpy()
    signal_preds = encodings[labels == 1]
    bkg_preds = encodings[labels == 0]

    threshold = np.quantile(signal_preds, 0.5)
    r_50 = len(bkg_preds) / (bkg_preds > threshold).sum()
    bkg_m = mass[labels == 0]
    bkg_passed = bkg_m[bkg_preds > threshold]
    bkg_rejected = bkg_m[bkg_preds < threshold]
    for n_bins in [10, 20, 50, 100]:
        hist1, bins = np.histogram(bkg_passed, bins=n_bins, density=True)
        hist2, _ = np.histogram(bkg_rejected, bins=bins, density=True)
        mx = (hist1 > 0) & (hist2 > 0)
        hist1 = hist1[mx]
        hist2 = hist2[mx]
        JSD = 0.5 * (entropy(hist1, 0.5 * (hist1 + hist2)) + entropy(hist2, 0.5 * (hist1 + hist2)))
        print(f'With {n_bins} bins. R_50: {r_50:.2f}, JSD_50: {1 / JSD:.2f}.')
    ####################################################################################################################


    if on_cluster():
        # plot random_cuts on both quantiles, cf-DNN and DNN
        plot_multi_cuts(data, results_directory / 'plots')

    # # Run bump hunts comparing to random cuts on the full data distribution
    # def run_rc(nm, discriminant, cuts=None, quantiles=None):
    #     rc_dir = results_directory / 'random_cuts' / f'{nm}'
    #     rc_dir.mkdir(exist_ok=True, parents=True)
    #     run_bump_hunt(discriminant, labels, mass, rc_dir, random_cuts=True, use_sideband=use_sideband, init_sig=0,
    #                   sub_dir='no_signal', cuts=cuts, quantiles=quantiles)
    #     run_bump_hunt(discriminant, labels, mass, rc_dir, random_cuts=True, use_sideband=use_sideband, init_sig=2,
    #                   sub_dir='two_sigma', cuts=cuts, quantiles=quantiles)
    #
    # run_rc('regressed', quantiles, cuts=quantiler.cuts, quantiles=tensor2numpy(quantiler.quantiles))
    # run_rc('quantiles', predictions)
    # run_rc('encodings', encodings)

    # # Run bump hunts using cuts on the discriminant to enhance the signal fraction and compare against true bkg
    # run_bump_hunt(quantiles, labels, mass, results_directory, random_cuts=False, use_sideband=use_sideband,
    #               cuts=quantiler.cuts, quantiles=tensor2numpy(quantiler.quantiles), sub_dir='regressed_bumps')
    # run_bump_hunt(predictions, labels, mass, quantile_dir, random_cuts=False, use_sideband=use_sideband)
    # run_bump_hunt(encodings, labels, mass, encodings_dir, random_cuts=False, use_sideband=use_sideband)
