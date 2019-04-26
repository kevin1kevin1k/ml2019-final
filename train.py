
# coding: utf-8

import argparse
from pathlib import Path
import numpy as np
import yaml
from copy import deepcopy
import sklearn.preprocessing
from scipy.special import logit, expit
import models
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid, GridSearchCV, cross_validate, KFold
from multioutput import MultiOutputRegressor # sklearn does not support 2D sample weight
from sklearn.metrics import mean_absolute_error, make_scorer
from datetime import datetime
import pandas as pd


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

np.random.seed(seed)


print('Loading data...')
data_dir = Path(data_path)
X_train = np.load(data_dir / 'X_train.npz')['arr_0']
X_test = np.load(data_dir / 'X_test.npz')['arr_0']
Y_train = np.load(data_dir / 'Y_train.npz')['arr_0']
n_targets = Y_train.shape[1]

X_train, unique_indices = np.unique(X_train, axis=0, return_index=True)
Y_train = Y_train[unique_indices]

if config['scaler_X']:
    scaler_X = getattr(sklearn.preprocessing, config['scaler_X'])()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
else:
    X_train_scaled = deepcopy(X_train)
    X_test_scaled = deepcopy(X_test)

Y_train_transformed = np.zeros_like(Y_train)

transform_dict = {
    'logit': logit,
    'logit_2y-1': lambda y: logit(2*y - 1),
    'log': np.log,
}
inv_transform_dict = {
    'logit': expit,
    'logit_2y-1': lambda y: (expit(y) + 1) / 2,
    'log': np.exp,
}

for i, transform in enumerate(config['transforms_Y']):
    Y_train_transformed[:, i] = transform_dict[transform](Y_train[:, i])

if config['scaler_Y']:
    scaler_Y = getattr(sklearn.preprocessing, config['scaler_Y'])()
    Y_train_scaled = scaler_Y.fit_transform(Y_train_transformed)
else:
    Y_train_scaled = deepcopy(Y_train_transformed)


model_name = config['model']
model_class = getattr(models, model_name)
param_grid = ParameterGrid(config['param_grid'])
params_formatter = config['params_formatter']
separate_targets = config['separate_targets']
sample_weight = config.get('sample_weight', None)
size = 100 if debug_mode else X_train_scaled.shape[0]

if sample_weight is not None:
    if sample_weight == 'target_reciprocal':
        sample_weight = 1 / Y_train
    else:
        raise ValueError('sample_weight {} is not supported'.format(sample_weight))
    sample_weight = sample_weight[:size]


def save_prediction(path, prediction, desc=None):
    if desc is None:
        desc = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    filename = Path(path) / 'prediction_{}.csv'.format(desc)
    np.savetxt(filename, prediction, delimiter=',', fmt='%.18f')


def scale_transform_clip(Y_scaled):
    if config['scaler_Y']:
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


def gridCV_and_predict():
    df = pd.DataFrame(columns=['params', 'WMAE', 'NAE'])
    param_grid_tqdm = tqdm(param_grid, desc=model_name)
    for i, params in enumerate(param_grid_tqdm):
        params_desc = params_formatter.format(**params)
        model_params = model_name + '_' + params_desc
        param_grid_tqdm.set_description(model_params)
        model = model_class(**params)
        if separate_targets:
            model = MultiOutputRegressor(model)

        model.fit(X_train_scaled[:size], Y_train_scaled[:size], sample_weight=sample_weight)
        Y_pred = scale_transform_clip(model.predict(X_test_scaled))
        save_prediction('predictions', Y_pred, model_params)

        cv_results = cross_validate(model, X_train_scaled[:size], Y_train_scaled[:size],
                                    scoring=score_funcs,
                                    cv=KFold(n_splits, random_state=seed),
                                    fit_params={'sample_weight': sample_weight})
        cv_errors = {f: -cv_results['test_'+f].mean() for f in ['WMAE', 'NAE']}
        cv_errors.update({'params': params_desc})
        df.loc[i] = cv_errors

    df.to_csv(Path('cv_results') / (model_name+'.csv'), index=False, float_format='%.6f')


print('Running grid CV...')
gridCV_and_predict()


# Used for finding the best params only
def find_best_params_and_predict(score_func):
    model = model_class()
    if separate_targets:
        model = MultiOutputRegressor(model)

    gs = GridSearchCV(model, param_grid=config['param_grid'],
                      scoring=make_scorer(score_func, greater_is_better=False),
                      cv=KFold(n_splits, random_state=seed),
                      n_jobs=1)

    gs.fit(X_train_scaled[:size], Y_train_scaled[:size], sample_weight=sample_weight)
    print('min error:', -gs.cv_results_['mean_test_score'][gs.best_index_])
    Y_pred = scale_transform_clip(gs.best_estimator_.predict(X_test_scaled))
    desc = model_name + '_' + params_formatter.format(**gs.best_params_)
    save_prediction('predictions', Y_pred, desc)

