
# coding: utf-8

# In[148]:


from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import argparse


# In[147]:


va_results = Path('va_results/')
va_predictions = Path('va_predictions/')
ensembles = Path('ensembles/')


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--error', type=str,
                        help='pick with WMAE or NAE for how=min', default='WMAE')
    parser.add_argument('-c', '--ensemble-config-path', type=str,
                        help='path to ensemble config file', default='ensemble_configs/001.yml')
    return parser.parse_args()


# In[ ]:


args = parse_args()
ensemble_config_path = args.ensemble_config_path
model_how_num = yaml.safe_load(open(ensemble_config_path))
blending = 'uniform' # validation
error = args.error


# In[143]:


Y_pred = np.zeros((2500, 3))
for target in range(3):
    error_target = '{}_{}'.format(error, target+1)
    for model, how, num in model_how_num:
        dfs = []
        for file in va_results.glob('{}*.csv'.format(model)):
            feature = file.stem.split('_')[-1]
            df = pd.read_csv(file)
            df['feature'] = feature
            dfs.append(df)
        df = pd.concat(dfs, sort=False)
        if how == 'random':
            df = df.sample(num, random_state=1126)
        elif how == 'min':
            df = df.sort_values(error_target)[:num]
        else:
            print('how error')

        preds = []
    #     errs = []
        for _, row in df.iterrows():
            pred_name = 'prediction_{}_{}_{}.csv'.format(row['model'], row['feature'], row['params'])
            pred_path = va_predictions / pred_name
            preds.append(pd.read_csv(pred_path, header=None)[target].values)
    #         errs.append(row[error_target])

        if blending == 'uniform':
            Y_pred[:, target] = np.average(preds, axis=0)
        else:
            print('blending error')

filename = ensembles / 'ensemble_{}_{}_{}.csv'.format(Path(ensemble_config_path).stem, blending, error)
np.savetxt(filename, Y_pred, delimiter=',', fmt='%.18f')

