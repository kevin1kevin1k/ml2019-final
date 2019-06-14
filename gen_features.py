
# coding: utf-8

# In[1]:


import numpy as np
from pathlib import Path


# In[4]:


class FeatureGen(object):
    def __init__(self):
        self.data_dir = Path('data')

        self.X_train = np.load(self.data_dir / 'X_train.npz')['arr_0']
        self.X_test = np.load(self.data_dir / 'X_test.npz')['arr_0']

        self.half = self.X_train.shape[1] // 2
        self.X_train_first_half = self.X_train[:, :self.half]
        self.X_test_first_half = self.X_test[:, :self.half]
        self.X_first_half = {'train': self.X_train_first_half, 'test': self.X_test_first_half}
        self.X_train_last_half = self.X_train[:, self.half:]
        self.X_test_last_half = self.X_test[:, self.half:]
        self.X_last_half = {'train': self.X_train_last_half, 'test': self.X_test_last_half}

    def gen_first_half(self, train_or_test):
        file_path = self.data_dir / 'X_{}_first_half.npz'.format(train_or_test)
        if not file_path.exists():
            print('Generating {} ...'.format(file_path))

            np.savez(file_path, self.X_first_half[train_or_test])

    def gen_first_half_stats(self, train_or_test):
        file_path = self.data_dir / 'X_{}_first_half_stats.npz'.format(train_or_test)
        if not file_path.exists():
            print('Generating {} ...'.format(file_path))

            X = self.X_first_half[train_or_test]
            self.X_first_half_stats = np.vstack([
                X.min(axis=1),
                X.max(axis=1),
                X.mean(axis=1),
                X.std(axis=1),
                (X[:, -1] - X[:, 0]) / self.half,
            ]).T
            np.savez(file_path, self.X_first_half_stats)

    def gen_first_half_deviation(self, train_or_test):
        file_path = self.data_dir / 'X_{}_first_half_deviation.npz'.format(train_or_test)
        if not file_path.exists():
            print('Generating {} ...'.format(file_path))

            X = self.X_first_half[train_or_test]
            self.X_first_half_deviation = np.zeros_like(X)
            for i in range(len(X)):
                slope = (X[i, -1] - X[i, 0]) / self.half
                self.X_first_half_deviation[i] = X[i, 0] + slope * np.arange(self.half) - X[i, :]
            np.savez(file_path, self.X_first_half_deviation)

    def gen_first_half_increment(self, train_or_test):
        file_path = self.data_dir / 'X_{}_first_half_increment.npz'.format(train_or_test)
        if not file_path.exists():
            print('Generating {} ...'.format(file_path))

            X = self.X_first_half[train_or_test]
            self.X_first_half_increment = X[:, 1:] - X[:, :-1]
            np.savez(file_path, self.X_first_half_increment)

    def gen_first_half_mean_per_100(self, train_or_test):
        file_path = self.data_dir / 'X_{}_first_half_mean_per_100.npz'.format(train_or_test)
        if not file_path.exists():
            print('Generating {} ...'.format(file_path))

            per = 100
            X = self.X_first_half[train_or_test]
            self.X_first_half_mean_per_100 = np.reshape(X, (-1, self.half // per, per)).mean(axis=-1)
            np.savez(file_path, self.X_first_half_mean_per_100)

    def gen_last_half(self, train_or_test):
        file_path = self.data_dir / 'X_{}_last_half.npz'.format(train_or_test)
        if not file_path.exists():
            print('Generating {} ...'.format(file_path))

            np.savez(file_path, self.X_last_half[train_or_test])

    def gen_last_half_group_min(self, train_or_test):
        file_path = self.data_dir / 'X_{}_last_half_group_min.npz'.format(train_or_test)
        if not file_path.exists():
            print('Generating {} ...'.format(file_path))

            n_groups = 50
            group_size = self.half // n_groups
            X = self.X_last_half[train_or_test]
            group_min = np.vstack([X[:, group_size*i + (i+1)] for i in range(n_groups)]).T
            np.savez(file_path, group_min)

    def gen_last_half_rfft_first500_20peaks(self, train_or_test):
        file_path = self.data_dir / 'X_{}_last_half_rfft_first500_20peaks.npz'.format(train_or_test)
        if not file_path.exists():
            print('Generating {} ...'.format(file_path))

            X = self.X_last_half[train_or_test]
            rfft = np.fft.rfft(X).real
            rfft_first500_20peaks = np.concatenate([rfft[:, 0:500:50], rfft[:, 50-1:500:50]], axis=1)
            np.savez(file_path, rfft_first500_20peaks)
    
    def gen_last_half_last_group(self, train_or_test):
        file_path = self.data_dir / 'X_{}_last_half_last_group.npz'.format(train_or_test)
        if not file_path.exists():
            print('Generating {} ...'.format(file_path))

            n_groups = 50
            group_size = self.half // n_groups
            X = self.X_last_half[train_or_test]
            last_group = X[:, -group_size:]
            np.savez(file_path, last_group)


# In[13]:


if __name__ == '__main__':
    feagen = FeatureGen()
    for gen in dir(feagen):
        if gen.startswith('gen'):
            for train_or_test in ['train', 'test']:
                getattr(feagen, gen)(train_or_test)

