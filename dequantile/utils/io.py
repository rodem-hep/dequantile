import os
import pathlib
import json
import time
import random

import numpy as np
import torch
import yaml


def on_cluster():
    """
    :return: True if running job on cluster
    """
    p = pathlib.Path().absolute()
    id = p.parts[:3][-1]
    if id == 'users':
        return True
    else:
        return False


def get_timestamp():
    formatted_time = time.strftime('%d-%b-%y||%H:%M:%S')
    return formatted_time


def get_log_root():
    directory = f'{get_top_dir()}/images/logs'
    os.makedirs(directory, exist_ok=True)
    return directory


def get_checkpoint_root(directory=None):
    if directory is None:
        directory = f'{get_top_dir()}/images/checkpoints'
    else:
        directory = f'{get_top_dir()}/images/{directory}'
    os.makedirs(directory, exist_ok=True)
    return directory


def get_data_root():
    return f'{get_top_dir()}/dequantile/data/downloads'


RESULTS_DIR = 'images'
if on_cluster():
    TOP_DIR = '/scratch/dequantile'
else:
    TOP_DIR = str(pathlib.Path().absolute())
LOG_DIR = 'logs'


def get_save_log_dirs(directory, name, log_dir=LOG_DIR):
    directory = pathlib.Path(TOP_DIR, RESULTS_DIR, directory, name)
    os.makedirs(directory, exist_ok=True)
    log_directory = pathlib.Path(TOP_DIR, LOG_DIR, log_dir)
    os.makedirs(log_directory, exist_ok=True)
    return directory, log_directory


class NoDataRootError(Exception):
    """Exception to be thrown when data root doesn't exist."""
    pass


def get_top_dir():
    p = pathlib.Path().absolute()
    id = p.parts[:3][-1]
    if id == 'samklein':
        sv_ims = '/Users/samklein/PycharmProjects/dequantile'
    elif id == 'users':
        sv_ims = '/home/users/k/kleins/MLproject/dequantile'
    else:
        # raise ValueError('Unknown path for saving images {}'.format(p))
        data_root_var = 'REPOROOT'
        try:
            return os.environ[data_root_var]
        except KeyError:
            raise NoDataRootError('Data root must be in environment variable {}, which'
                                  ' doesn\'t exist.'.format(data_root_var))
    return sv_ims


def get_data_root():
    # if on_cluster():
    #     return '/scratch'
    # else:
    return f'{get_top_dir()}/dequantile/data/downloads'


def get_image_data_root(name):
    return f'{get_data_root()}/images/{name}'


def register_experiment(path, args):
    """
    Save the arguments used to run an experiment.
    :param path: A Directory type object.
    :param args: the output of argparse used to save the data
    :return: None
    """
    dump_dict(path, vars(args))


def dump_dict(path, dictionary):
    with open(f'{path}.yml', 'w') as outfile:
        yaml.dump(dictionary, outfile, default_flow_style=False)


class DotDict(dict):
    """ot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_experiment(directory):
    """
    Read the arguments used to run an experiment.
    :param directory: Path to exp_info.yml file.
    :return: args: the output of argparse used in sb_to_sb.py as
    DotDict object.
    """
    with open(f'{directory}', 'r') as stream:
        config_dict = yaml.safe_load(stream)
    return DotDict(config_dict)


def reset_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
