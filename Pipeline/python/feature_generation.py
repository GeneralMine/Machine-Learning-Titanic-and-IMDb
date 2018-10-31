import numpy as np
import pandas as pd

def generate_features(features, train_X, test_X):
    trainsize = len(train_X)
    data = train_X.append(test_X, ignore_index=True)
    
    for func in features:
        feature, name = func(data)
        add_feature(feature, name, data)
    
    return data[:trainsize], data[trainsize:].reset_index(drop=True)

def generate_Y_dependent_features(features, train_X, train_Y, test_X):
    for func in features:
        feature_train, feature_test, name = func(train_X, train_Y, test_X)
        add_ydependent_feature(feature_train, feature_test, train_X, test_X, name)
    return train_X, test_X

def add_feature(feature, name, data):
    #eventuell noch type check
    if not len(feature) == len(data):
        print('Error: expected feature "' + name + '" to be of length ' + len(data) + ' but was ' + len(feature))
        return
    if feature.isnull().any():
        print('Error: feature "' + name + '" contains nan')
        return
    data[name] = feature

def add_ydependent_feature(feature_train, feature_test, train_X, test_X, name):
    if not len(feature_train) == len(train_X):
        print('Error: expected feature "' + name + '" to be of length ' + len(train_X) + ' but was ' + len(feature_train))
        return
    if not len(feature_test) == len(test_X):
        print('Error: expected feature "' + name + '" to be of length ' + len(test_X) + ' but was ' + len(feature_test))
        return
    if feature_train.isnull().any() or feature_test.isnull().any():
        print('Error: feature "' + name + '" contains nan')
        return
    train_X[name] = feature_train
    test_X[name] = feature_test
    
drop_after_feature_generation = ['name','ticket','cabin','home.dest','boat']

def drop_non_numeric_columns(train_X, test_X):
    for col in drop_after_feature_generation:
        train_X.drop(col, axis=1, inplace=True)
        test_X.drop(col, axis=1, inplace=True)
    return train_X, test_X