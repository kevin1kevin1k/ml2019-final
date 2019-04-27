
# coding: utf-8

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from copy import deepcopy
from tqdm import tqdm
from itertools import product
from datetime import datetime
from scipy.special import logit, expit
import sklearn.preprocessing
from sklearn.model_selection import ParameterGrid, GridSearchCV, cross_validate, KFold
from sklearn.metrics import mean_absolute_error, make_scorer

import models
from multioutput import MultiOutputRegressor # sklearn does not support 2D sample weight


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--ipykernal-file',
                        help='for compatibility in Jupyter')
    parser.add_argument('-c', '--config-path', type=str,
                        help='path to config file', default='configs/config_Ridge.yml')
    parser.add_argument('-d', '--data-path', type=str,
                        help='path to data sets', default='data')
    parser.add_argument('-s', '--cv-splits', type=int,
                        help='number of splits for CV', default=5)
    parser.add_argument('-r', '--random-seed', type=int,
                        help='random seed for numpy and CV', default=1126)
    parser.add_argument('-b', '--debug-mode', type=str2bool,
                        help='faster debugging by using only the first 100 training data', default=True)
    args = parser.parse_args()
    
    return {
        'config': yaml.safe_load(open(args.config_path)),
        'data_path': args.data_path,
        'n_splits': args.cv_splits,
        'seed': args.random_seed,
        'debug_mode': args.debug_mode
    }


args = parse_args()
config = args['config']
data_path = args['data_path']
n_splits = args['n_splits']
seed = args['seed']
debug_mode = args['debug_mode']

predictions_path = 'predictions'
cv_results_path = 'cv_results'

np.random.seed(seed)


print('Loading data...')
data_dir = Path(data_path)
X_train = np.load(data_dir / 'X_train.npz')['arr_0']
X_test = np.load(data_dir / 'X_test.npz')['arr_0']
Y_train = np.load(data_dir / 'Y_train.npz')['arr_0']
n_targets = Y_train.shape[1]

X_train, unique_indices = np.unique(X_train, axis=0, return_index=True)
Y_train = Y_train[unique_indices]

if 'scaler_X' in config:
    scaler_X = getattr(sklearn.preprocessing, config['scaler_X'])()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
else:
    X_train_scaled = deepcopy(X_train)
    X_test_scaled = deepcopy(X_test)

Y_train_transformed = np.zeros_like(Y_train)

transform_dict = {
    'id': lambda y: y,
    'logit': logit,
    'logit_2y-1': lambda y: logit(2*y - 1),
    'log': np.log,
}
inv_transform_dict = {
    'id': lambda y: y,
    'logit': expit,
    'logit_2y-1': lambda y: (expit(y) + 1) / 2,
    'log': np.exp,
}

for i, transform in enumerate(config['transforms_Y']):
    Y_train_transformed[:, i] = transform_dict[transform](Y_train[:, i])

if 'scaler_Y' in config:
    scaler_Y = getattr(sklearn.preprocessing, config['scaler_Y'])()
    Y_train_scaled = scaler_Y.fit_transform(Y_train_transformed)
else:
    Y_train_scaled = deepcopy(Y_train_transformed)


model_name = config['model']
model_class = getattr(models, model_name)
param_grid = ParameterGrid(config['param_grid'])
fit_param_grid = ParameterGrid(config.get('fit_param_grid', {}))
params_formatter = config['params_formatter']
separate_targets = config.get('separate_targets', False)
size = 100 if debug_mode else X_train_scaled.shape[0]


def save_prediction(prediction, desc=None):
    if desc is None:
        desc = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if debug_mode:
        desc += '_debug'
    filename = Path(predictions_path) / 'prediction_{}.csv'.format(desc)
    np.savetxt(filename, prediction, delimiter=',', fmt='%.18f')


def scale_transform_clip(Y_scaled):
    if 'scaler_Y' in config:
        Y_transformed = scaler_Y.inverse_transform(Y_scaled)
    else:
        Y_transformed = Y_scaled

    Y = np.zeros_like(Y_transformed)
    for i, transform in enumerate(config['transforms_Y']):
        Y[:, i] = inv_transform_dict[transform](Y_transformed[:, i])

    return np.clip(Y, [0, 0, 0.5], [1, float('inf'), 1])


class Error():
    def WMAE(Y_true, Y_pred):
        Y_true, Y_pred = map(scale_transform_clip, (Y_true, Y_pred))
        weights = [200, 1, 300]
        e = mean_absolute_error(Y_true, Y_pred, multioutput=weights)
        return e * sum(weights)

    def NAE(Y_true, Y_pred):
        Y_true, Y_pred = map(scale_transform_clip, (Y_true, Y_pred))
        sample_weight = 1 / Y_true
        e = mean_absolute_error(Y_true, Y_pred, sample_weight=sample_weight)
        return e

score_funcs = {f: make_scorer(getattr(Error, f), greater_is_better=False) for f in ['WMAE', 'NAE']}


def parse_fit_params(fit_params):
    parsed = dict(fit_params)
    sample_weight_desc = fit_params.get('sample_weight', None)
    if sample_weight_desc is not None:
        if sample_weight_desc == 'uniform':
            sample_weight = None
        elif sample_weight_desc == 'target_reciprocal':
            sample_weight = (1 / Y_train)[:size]
        else:
            raise ValueError('sample_weight {} is not supported'.format(sample_weight))
        parsed['sample_weight'] = sample_weight

    return parsed

def gridCV_and_predict():
    csv_filename = model_name + ('_debug' if debug_mode else '') + '.csv'
    csv_path = Path(cv_results_path) / csv_filename
    done_params = pd.read_csv(csv_path)['params'].tolist() if csv_path.exists() else []

    param_fit_param_grid = tqdm(product(param_grid, fit_param_grid), desc=model_name)
    for params, fit_params in param_fit_param_grid:
        params_desc = params_formatter.format(**params, **fit_params)
        if params_desc in done_params:
            continue
        model_params = model_name + '_' + params_desc
        param_fit_param_grid.set_description(model_params)
        model = model_class(**params)
        if separate_targets:
            model = MultiOutputRegressor(model)

        fit_params_ = parse_fit_params(fit_params)
        model.fit(X_train_scaled[:size], Y_train_scaled[:size], **fit_params_)
        Y_pred = scale_transform_clip(model.predict(X_test_scaled))
        save_prediction(Y_pred, model_params)

        cv_results = cross_validate(model, X_train_scaled[:size], Y_train_scaled[:size],
                                    scoring=score_funcs,
                                    cv=KFold(n_splits, random_state=seed),
                                    fit_params=fit_params_)
        cv_errors = {f: -cv_results['test_'+f].mean() for f in ['WMAE', 'NAE']}
        cv_errors.update({'params': params_desc})
        df = pd.DataFrame(columns=['params', 'WMAE', 'NAE'])
        df.loc[0] = cv_errors
        df.to_csv(csv_path, index=False, float_format='%.6f', mode='a', header=not csv_path.exists())


print('Running grid CV...')
gridCV_and_predict()

