
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
from sklearn.model_selection import ParameterGrid, train_test_split
import pickle

import models
from multioutput import MultiOutputRegressor # sklearn does not support 2D sample weight
from gen_features import FeatureGen


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
    parser.add_argument('-e', '--feature-config-path', type=str,
                        help='path to feature config file', default='feature_configs/features_001.yml')
    parser.add_argument('-v', '--save-model', type=str2bool,
                        help='whether to save the models with best CV scores', default=False)
    parser.add_argument('-u', '--unique', type=str2bool,
                        help='whether to remove duplicate training data', default=False)
    parser.add_argument('-r', '--random-seed', type=int,
                        help='random seed for numpy and CV', default=1126)
    parser.add_argument('-b', '--debug-mode', type=str2bool,
                        help='faster debugging by using only the first 100 training data', default=True)
    return parser.parse_args()


args = parse_args()
config = yaml.safe_load(open(args.config_path))
feature_config_path = args.feature_config_path
feature_id = feature_config_path[-len('000.yml'):-len('.yml')]
feature_config = yaml.safe_load(open(feature_config_path))
data_path = args.data_path
save_model = args.save_model
seed = args.random_seed
debug_mode = args.debug_mode
unique = args.unique

predictions_path = 'va_predictions'
va_results_path = 'va_results'

np.random.seed(seed)


model_name = config['model']
model_class = getattr(models, model_name)
param_grid = ParameterGrid(config['param_grid'])
fit_param_grid = ParameterGrid(config.get('fit_param_grid', {}))
params_formatter = config['params_formatter']
error_names = config.get('error_funcs', ['WMAE', 'NAE'])


print('Loading data...')
data_dir = Path(data_path)
feagen = None

def load_features(train_or_test):
    global feagen

    features = []
    for feature_name in feature_config['features']:
        feature_filename = 'X_{}_{}.npz'.format(train_or_test, feature_name)
        feature_path = data_dir / feature_filename
        if not feature_path.exists():
            if feagen is None:
                feagen = FeatureGen()
            getattr(feagen, 'gen_{}'.format(feature_name))(train_or_test)
        feature = np.load(feature_path)['arr_0']
        features.append(feature)
    return np.concatenate(features, axis=1)

X_train = load_features('train')
X_test = load_features('test')
Y_train = np.load(data_dir / 'Y_train.npz')['arr_0']
n_targets = Y_train.shape[1]

if unique:
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

size = 100 if debug_mode else X_train_scaled.shape[0]
va_size = 10 if debug_mode else 2500
X_train_scaled = X_train_scaled[:size]
Y_train_scaled = Y_train_scaled[:size]
X_tr, X_va, Y_tr, Y_va = train_test_split(X_train_scaled, Y_train_scaled,
                                          test_size=va_size, shuffle=False)
print('Finish!')


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
    def _AE(Y_true, Y_pred, weighted=False, normalized=False):
        Y_true, Y_pred = map(scale_transform_clip, (Y_true, Y_pred))
        denom = Y_true if normalized else 1
        errors = np.sum(np.abs(Y_true - Y_pred) / denom, axis=0) / len(Y_true)
        weights = [300, 1, 200] if weighted else [1, 1, 1]
        return errors @ weights, errors

    def WMAE(Y_true, Y_pred):
        return Error._AE(Y_true, Y_pred, weighted=True)

    def NAE(Y_true, Y_pred):
        return Error._AE(Y_true, Y_pred, normalized=True)

error_funcs = {f: getattr(Error, f) for f in error_names}


# a = np.array([[1., 2., 6.], [3., 3., 3.]])
# b = np.array([[0., 0., 0.], [0., 0., 0.]])

# for error_name, error_func in error_funcs.items():
#     print(error_name, error_func(a, b))


def parse_fit_params(fit_params, va=False):
    parsed = dict(fit_params)
    sample_weight_desc = fit_params.get('sample_weight', None)
    if sample_weight_desc is not None:
        if sample_weight_desc == 'uniform':
            sample_weight = None
        elif sample_weight_desc == 'target_reciprocal':
            sample_weight = (1 / Y_train)[:size]
        else:
            raise ValueError('sample_weight {} is not supported'.format(sample_weight))
        if sample_weight is not None and va:
            sample_weight = sample_weight[:-va_size]
        parsed['sample_weight'] = sample_weight

    return parsed


def grid_and_predict():
    csv_filename = '{}_{}{}.csv'.format(model_name, feature_id, ('_debug' if debug_mode else ''))
    csv_path = Path(va_results_path) / csv_filename
    done_params = pd.read_csv(csv_path)['params'].tolist() if csv_path.exists() else []

    param_fit_param_grid = tqdm(product(param_grid, fit_param_grid), desc=model_name)
    for params, fit_params in param_fit_param_grid:
        params_desc = params_formatter.format(**params, **fit_params)
        if params_desc in done_params:
            continue

        param_fit_param_grid.set_description(params_desc)
        fit_params_ = parse_fit_params(fit_params, va=False)
        model = MultiOutputRegressor(model_class(**params))
        model.fit(X_train_scaled, Y_train_scaled, **fit_params_)
        Y_pred = scale_transform_clip(model.predict(X_test_scaled))
        desc = '{}_{}_{}'.format(model_name, feature_id, params_desc)
        save_prediction(Y_pred, desc)
        if save_model:
            pickle.dump(model, open('models/{}.pkl'.format(desc), 'wb'))

        fit_params_ = parse_fit_params(fit_params, va=True)
        model = MultiOutputRegressor(model_class(**params))
        model.fit(X_tr, Y_tr, **fit_params_)
        Y_pred = model.predict(X_va)

        columns = [f + target for f in error_names
                              for target in [''] + ['_' + str(i+1) for i in range(n_targets)]]
        va_results = {}
        for error_name, error_func in error_funcs.items():
            error, errors = error_func(Y_va, Y_pred)
            va_results[error_name] = error
            for i in range(n_targets):
                va_results['{}_{}'.format(error_name, i+1)] = errors[i]
        va_results.update({'model': model_name, 'params': params_desc})
        df = pd.DataFrame(columns=['model', 'params']+columns)
        df.loc[0] = va_results
        df.to_csv(csv_path, index=False, float_format='%.6f', mode='a', header=not csv_path.exists())


print('Running grid...')
grid_and_predict()
print('Finish!')

