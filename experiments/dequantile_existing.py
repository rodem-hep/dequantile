import argparse
import numpy as np
import pandas as pd
import pathlib

import torch
from matplotlib import pyplot as plt
from nflows.utils import tensor2numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from dequantile.data.boosted_w_data import BoostedW
from dequantile.models.MLP import MLP
from dequantile.models.classifiers.base import Classifier
from dequantile.models.classifiers.disco import DisCoClassifier
from dequantile.models.classifiers.mode import MoDeClassifier
from dequantile.models.classifiers.networks import MoDeModel
from dequantile.models.flows.flow_decorrelator import FlowDecorrelatorClassifier, ConditionalFeatureDecorrelator, \
    FlowDecorrelator
from dequantile.models.flows.inns import ClassifierInn, InnEnsemble
from dequantile.models.quantiles.pinball import QuantileRegressor
from dequantile.models.training import train
from dequantile.utils import io
from dequantile.utils.evaluation import run_evaluation
from dequantile.utils.metrics import plot_mode_metrics
from dequantile.utils.torch_utils import no_more_grads, MSELoss, TupleDataset


def parse_args():
    # TODO filter args for essentials
    parser = argparse.ArgumentParser()

    # Saving
    parser.add_argument('-d', '--outputdir', type=str, default='disco_decor',
                        help='Choose the base output directory')
    parser.add_argument('-n', '--outputname', type=str, default='local',
                        help='Set the output name directory')
    parser.add_argument('--log_dir', type=str, default='local',
                        help='Set the output name directory')
    parser.add_argument('--seed', type=int, default=0,
                        help='Add this to 42 to set the seed.')

    # Training
    parser.add_argument('--dec_inputs', type=int, default=0,
                        help='1 = decorrelate inputs directly, 0 = train on correlated features.')
    # Classifier
    parser.add_argument('--cf_epochs', type=int, default=1,
                        help='Number of epochs to train both the classifier and the flow.')
    parser.add_argument('--cf_bs', type=int, default=256,
                        help='Batch size for the classifier training.')
    parser.add_argument('--cf_lr', type=float, default=0.001,
                        help='Learning rate for the classifier.')
    parser.add_argument('--reload', type=int, default=0,
                        help='Reload saved models from last checkpoint.')

    # Flow
    parser.add_argument('--flow_epochs', type=int, default=10,
                        help='Number of epochs to train the flow after the classifier is finished.')
    parser.add_argument('--flow_bs', type=int, default=256,
                        help='Batch size for the flow training.')
    parser.add_argument('--flow_lr', type=float, default=0.001,
                        help='Learning rate for the flow.')
    parser.add_argument('--quantiles', type=str, default='0.5,0.9,0.95,0.99',
                        help='Comma separated list of the quantiles to regress.')

    # Quantile regressor
    parser.add_argument('--reg_epochs', type=int, default=10,
                        help='Number of epochs to train the quantile regressor after the classifier is finished.')

    # Model args
    parser.add_argument('--n_knots', type=int, default=10,
                        help='Number of knots to use in the flow.')
    parser.add_argument('--f_nodes', type=int, default=64,
                        help='Number of hidden layers to use in the network that learns the knot placements.')
    parser.add_argument('--n_stack', type=int, default=1,
                        help='The number of sub inns that make up the INN.')
    parser.add_argument('--n_ensemble', type=int, default=1,
                        help='The number of INNs to ensemble.')
    parser.add_argument('--r_nodes', type=int, default=64,
                        help='The number of nodes to use in the quantile regressor.')

    # Meta model args
    parser.add_argument('--decor_method', type=int, default=2,
                        help='2 = train with DisCo decorrelation,'
                             '1 = train with mode decorrelation,'
                             '0 = train on dequantile.')
    parser.add_argument('--drop_mass', type=int, default=1,
                        help='1 = drop mass as a training feature, 0 = keep mass as a training feature.')
    parser.add_argument('--drop_pt', type=int, default=0,
                        help='1 = drop p_T as a training feature, 0 = keep p_T as a training feature.')
    parser.add_argument('--mode_model', type=int, default=1,
                        help='1 = same cf set up as mode paper, 0 = unoptimised set up.')
    parser.add_argument('--extension', type=str, default='0.0',
                        help='Alpha used to balance the mode loss.')
    parser.add_argument('--mode_order', type=int, default=0,
                        help='Order to use in the mode loss.')
    parser.add_argument('--bce', type=int, default=1,
                        help='1 = binary cross entropy, 0 = mean squared error')
    parser.add_argument('--resample_mass', type=int, default=0,
                        help='0 = no resampling, 1 = constant s/b ratio in all bins, 2 = flat mass profile, '
                             '3 = constant s/b and flat mass ratio.')
    parser.add_argument('--resample_decor', type=int, default=0,
                        help='0 = no resampling, 1 = flatten mass spectrum (for decorrelator training).')
    parser.add_argument('--use_weights', type=int, default=2,
                        help='Use class weights for the classifier.')

    # Bump hunt args
    parser.add_argument('--use_sideband', type=int, default=0)

    return parser.parse_args()


def boosted_w_classification():
    args = parse_args()
    results_directory, log_dir = io.get_save_log_dirs(args.outputdir, args.outputname, args.log_dir)
    plots_dir = results_directory / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_directory / 'saved_models'
    io.register_experiment(results_directory / 'exp_info', args)
    io.reset_seed(42 + args.seed)

    # Set the theme
    plt.style.use('seaborn-ticks')

    ####################################################################################################################
    ####################################################################################################################
    # Get the data
    # Use the MinMaxScaler as in MoDe unless directly decorrelating the inputs with a flow
    mass_preprocessor = MinMaxScaler()
    # Disco has an unfortunate effect of not spreading scores between 0, 1 but has support on some other range.
    # Not scaling to [0,1] ensures test data is still within tail bounds
    score_preprocessor = MinMaxScaler(feature_range=(0.05, 0.95))
    data_dir = pathlib.Path('/load_dir')
    splits = ['train', 'val', 'test']
    data_dict = {file: data_dir / f'{file}_{args.extension}.npy' for file in splits}

    def get_dataset(nm, data, mass_preprocessor, score_preprocessor):
        if nm == 'train':
            mass_preprocessor.fit(data[:, -1:])
            score_preprocessor.fit(data[:, :1])
        if nm == 'test':
            # This was forgotten in the disco code dump. Corrected here.
            data[:, -1:] = data[:, -1:] * 250 + 50
        data[:, -1:] = mass_preprocessor.transform(data[:, -1:])
        data[:, :1] = score_preprocessor.transform(data[:, :1])
        data, label, mass = torch.Tensor(data.transpose())
        if nm in ['train', 'val']:
            mx = label == 0
            data = data[mx]
            mass = mass[mx]
            label = label[mx]
        data, mass = [dt.view(-1, 1) for dt in [data, mass]]
        label = label.view(-1)
        weight = torch.ones_like(mass)
        return TupleDataset(data, label, mass, weight)

    for nm, file in data_dict.items():
        with open(file, 'rb') as f:
            np_data = np.load(f)
        data_dict[nm] = get_dataset(nm, np_data, mass_preprocessor, score_preprocessor)

    ####################################################################################################################
    ####################################################################################################################
    # Define evaluation hyperparameters
    valid_batch_size = 5000

    data_loaders = {}
    for nm, dt in data_dict.items():
        data_loaders[nm] = DataLoader(dt, shuffle=True, batch_size=args.flow_bs if nm == 'train' else valid_batch_size)

    # Device things
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####################################################################################################################
    # Define a flow to learn the conditional distribution of the background labels
    def cinn():
        return ClassifierInn(args.f_nodes, args.n_knots, args.n_stack)

    inn = InnEnsemble(cinn, args.n_ensemble)
    flow = FlowDecorrelator(inn).to(device)

    ####################################################################################################################
    # TODO do quantile regression for this as well
    # # Define a model for regressing quantiles
    # quantiles = [float(q) for q in args.quantiles.split(',')]
    # quantile_network = MLP(1, len(quantiles), N=args.r_nodes)
    # quantile_regressor = QuantileRegressor(classifier, quantile_network, quantiles=quantiles).to(device)

    ####################################################################################################################
    ####################################################################################################################
    if args.reload:
        flow.load_state_dict(
            torch.load(models_dir / f'flow_decorrelator/{args.flow_epochs - 1}', map_location=device))
        # quantile_regressor.load_state_dict(
        #     torch.load(models_dir / f'quantile_regressor/{args.flow_epochs - 1}', map_location=device))
    else:

        ################################################################################################################
        if args.flow_epochs > 0:
            # Train the flow to learn the background distribution and turn of the resampling
            train(flow, data_loaders['train'], args.flow_epochs, args.flow_lr, device, models_dir / 'flow_decorrelator',
                  sv_nm=plots_dir / 'flow_loss.png', decorrelator=False, valid_loader=data_loaders['val'])

        # ################################################################################################################
        # # Train a quantile regressor to predict specific quantiles
        # train(quantile_regressor, data_object, args.reg_epochs, args.flow_lr, device, models_dir / 'quantile_regressor',
        #       sv_nm=plots_dir / 'regressor_loss.png', decorrelator=resample)

    ####################################################################################################################
    ####################################################################################################################

    enc = []
    predictions = []
    labels = []
    mass = []
    with torch.no_grad():
        for t_step, (encodings, t_label, t_mass, _) in enumerate(data_loaders['test']):
            encodings = encodings.to(device)
            enc += [encodings]
            predictions += [flow.transform_to_noise(encodings, t_mass.view(-1, 1).to(device))[0]]
            labels += [t_label]
            mass += [t_mass]
    encodings = torch.cat(enc, dim=0).view(-1, 1)
    predictions = torch.cat(predictions, dim=0).view(-1, 1)
    labels = torch.cat(labels, dim=0).view(-1, 1)
    mass = torch.cat(mass, dim=0).cpu().numpy().reshape(-1, 1)

    # Dump everything to hdf as a dataframe
    data = pd.DataFrame({
        'mass': mass.reshape(-1),
        'encodings': tensor2numpy(encodings).reshape(-1),
        'labels': tensor2numpy(labels).reshape(-1),
        'predictions': tensor2numpy(predictions).reshape(-1),
    })
    data.to_hdf(results_directory / 'results_df.h5', key='df', index=False)

    corr_dir = results_directory / 'mode_metrics'
    corr_dir.mkdir(exist_ok=True, parents=True)
    d_jsd, dr50, dpJSDs = plot_mode_metrics(predictions, labels, mass, corr_dir / 'quantiles.png')
    c_jsd, cr50, cpJSDs = plot_mode_metrics(encodings, labels, mass, corr_dir / 'encodings.png')
    print(f'Original JSD, R_50 :{c_jsd, cr50}')
    print(f'Dequantiled JSD, R_50 :{d_jsd, dr50}')
    np.save(corr_dir / 'mode_metrics.npy', np.array([d_jsd, dr50, c_jsd, cr50]))


if __name__ == '__main__':
    boosted_w_classification()
