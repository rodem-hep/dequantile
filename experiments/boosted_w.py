import argparse
import pathlib

import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn

from dequantile.data.boosted_w_data import BoostedW
from dequantile.models.MLP import MLP
from dequantile.models.classifiers.base import Classifier
from dequantile.models.classifiers.disco import DisCoClassifier
from dequantile.models.classifiers.mode import MoDeClassifier
from dequantile.models.classifiers.networks import MoDeModel
from dequantile.models.flows.flow_decorrelator import FlowDecorrelatorClassifier, ConditionalFeatureDecorrelator
from dequantile.models.flows.inns import ClassifierInn, InnEnsemble
from dequantile.models.quantiles.pinball import QuantileRegressor
from dequantile.models.training import train
from dequantile.utils import io
from dequantile.utils.evaluation import run_evaluation
from dequantile.utils.torch_utils import no_more_grads, MSELoss


def parse_args():
    parser = argparse.ArgumentParser()

    # Saving
    parser.add_argument('-d', '--outputdir', type=str, default='boosted_w',
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
    parser.add_argument('--cf_epochs', type=int, default=10,
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
    parser.add_argument('--alpha', type=float, default=300,
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
    if args.dec_inputs:
        preprocessor = StandardScaler()
        center_mass = False
    else:
        preprocessor = MinMaxScaler()
        center_mass = True
    data_object = BoostedW(preprocessor, unblind=True, drop_mass=args.drop_mass, drop_pt=args.drop_pt,
                           resample=args.resample_mass, use_weights=args.use_weights, center_mass=center_mass)

    ####################################################################################################################
    ####################################################################################################################
    # Define evaluation hyperparameters
    valid_batch_size = 5000

    # Device things
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####################################################################################################################
    # Define a feature decorrelating network
    if args.dec_inputs:
        # If decorrelating the input the data is expected to be scaled between [-3, 3]
        # data_object.scale = 3
        encoder = ConditionalFeatureDecorrelator(data_object.data_dim, 3.).to(device)
    else:
        encoder = None

    ####################################################################################################################
    # Define a dense net to do classification
    if args.mode_model == 1:
        network = MoDeModel(data_object.data_dim)
    elif args.mode_model == 2:
        network = MoDeModel(data_object.data_dim, activation="relu")
    else:
        network = MLP(data_object.data_dim, 1, N=64, activation=nn.Sigmoid())
    primary_loss = None if args.bce else MSELoss()
    classifier = Classifier(network, primary_loss=primary_loss, encoder=encoder).to(device)
    if args.decor_method == 1:
        # This is the default setting for MoDe
        args.cf_bs = int(2 ** 14)
        classifier = MoDeClassifier(network, args.mode_order, alpha=args.alpha, primary_loss=primary_loss).to(device)
    elif args.decor_method == 2:
        # This is the default setting for DisCo
        args.cf_bs = 2048
        classifier = DisCoClassifier(network, alpha=args.alpha, primary_loss=primary_loss).to(device)

    ####################################################################################################################
    # Define a flow to learn the conditional distribution of the background labels
    def cinn():
        return ClassifierInn(args.f_nodes, args.n_knots, args.n_stack)

    inn = InnEnsemble(cinn, args.n_ensemble)
    flow = FlowDecorrelatorClassifier(classifier, inn).to(device)

    ####################################################################################################################
    # Define a model for regressing quantiles
    quantiles = [float(q) for q in args.quantiles.split(',')]
    quantile_network = MLP(1, len(quantiles), N=args.r_nodes)
    quantile_regressor = QuantileRegressor(classifier, quantile_network, quantiles=quantiles).to(device)

    ####################################################################################################################
    ####################################################################################################################
    if args.reload:
        if args.dec_inputs:
            encoder.load_state_dict(
                torch.load(models_dir / f'input_deccorelator/{args.cf_epochs - 1}', map_location=device))
        classifier.load_state_dict(
            torch.load(models_dir / f'classifiers/{args.cf_epochs - 1}', map_location=device))
        flow.load_state_dict(
            torch.load(models_dir / f'flow_decorrelator/{args.flow_epochs - 1}', map_location=device))
        quantile_regressor.load_state_dict(
            torch.load(models_dir / f'quantile_regressor/{args.flow_epochs - 1}', map_location=device))
    else:
        # If there is a feature mass decorrelator then train it
        if args.dec_inputs:
            # Define batch_sizes and drop the signal part
            data_object.setup_loaders(args.cf_bs, valid_batch_size, bkg_only=True)
            train(encoder, data_object, args.cf_epochs, args.cf_lr, device, models_dir / 'input_deccorelator',
                  sv_nm=plots_dir / 'feature_decorrelator_loss.png')
            # Turn off the encoder weights
            no_more_grads(encoder)

        ################################################################################################################
        # Train the classifier
        # Define batch_sizes and include signal
        data_object.setup_loaders(args.cf_bs, valid_batch_size, bkg_only=False)
        train(classifier, data_object, args.cf_epochs, args.cf_lr, device, models_dir / 'classifiers',
              sv_nm=plots_dir / 'cf_loss.png')
        # Turn off the grads of the classifier
        no_more_grads(classifier)

        ################################################################################################################
        # Train the flow to learn the background distribution and turn of the resampling
        data_object.setup_loaders(args.flow_bs, valid_batch_size, bkg_only=True)
        resample = 2 if args.resample_decor else 0
        data_object.update_sampler(resample)
        train(flow, data_object, args.flow_epochs, args.flow_lr, device, models_dir / 'flow_decorrelator',
              sv_nm=plots_dir / 'flow_loss.png', decorrelator=resample)

        ################################################################################################################
        # Train a quantile regressor to predict specific quantiles
        train(quantile_regressor, data_object, args.reg_epochs, args.flow_lr, device, models_dir / 'quantile_regressor',
              sv_nm=plots_dir / 'regressor_loss.png', decorrelator=resample)

    ####################################################################################################################
    ####################################################################################################################
    # Evaluate on the full sets, not just background
    data_object.bkg_only = False
    data_object.update_sampler(0)
    train_loader, valid_loader, test_loader = data_object.get_loaders()

    # Evaluate the mass decorrelation performance and perform a bump hunt
    run_evaluation(classifier, flow, quantile_regressor, test_loader, device, data_object.unscale_mass,
                   results_directory, use_sideband=args.use_sideband)


if __name__ == '__main__':
    boosted_w_classification()
