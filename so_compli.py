import os
import csv
import numpy as np
import pickle

current_directory = os.path.dirname(__file__)
data_directory = os.path.join(current_directory, 'data')
data_directory = "C:\so_compli\data"

def load_data(prefix='train'):
    train_path = os.path.join(data_directory, f'{prefix}.csv')

    pkl_path = os.path.join(data_directory, f'{prefix}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
    else :
        raw_data = np.loadtxt(train_path, skiprows=1, dtype='int', delimiter=',')
        with open(pkl_path, 'wb') as f:
            pickle.dump(raw_data, f)
    return raw_data

def load_train_data():
    return load_data(prefix='train')

def load_test_data():
    return load_data(prefix='test')

if __name__ == '__main__':
    load_train_data()