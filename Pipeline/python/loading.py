import numpy as np
import pandas as pd
from os.path import isfile
import math
from sklearn.preprocessing import LabelEncoder

def load_data(filename, seperator = ',', col_names = None):
	if not isfile(filename):
		return pd.DataFrame()
	else:
		return pd.read_csv(filename, sep = seperator, names = col_names)

def load_training_data(use_cached=True, benchmark='random'):
    data = load_data('data/titanic.csv')
    if benchmark == 'random':
        mask = random_mask(len(data))
    else:
        mask = np.loadtxt('data/' + benchmark).astype(bool)
    train, test = split_train_test(data, mask)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train_X, train_Y = splitXY(train)
    test_X, test_Y = splitXY(test)
    return train_X, train_Y, test_X, test_Y

default_Y_columns = ['survived']

def random_mask(size):
    mask = np.zeros(size)
    mask[:150] = 1
    return np.random.permutation(mask).astype(bool)

def split_train_test(data, mask):
    test = data[mask]
    train = data[~mask]
    return train, test
    
def dump_data(data, filename, seperator = ','):
	data.to_csv(filename, index = False, sep = seperator)
    
def shuffle(data):
    return data.reindex(np.random.permutation(data.index))

def splitXY(data, Y_columns=default_Y_columns):
	Y = data[Y_columns]
	X = data.drop(Y_columns, axis=1)
	return X, Y
