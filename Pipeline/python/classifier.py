import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# use own Classifier class to define interface for classifier
# that enables us to use classifier that are not part of sklear without assuming a certain interface
# in addition we can encapsulate further steps like parameter selection in the train method


class Classifier:

    def __init__(self, clf=None, name=None):
        self._name = name
        self._clf = clf

    def train(self, X: pd.DataFrame, Y: pd.DataFrame):
        self._clf.fit(X, Y)

    def predict(self, X: pd.DataFrame):
        return self._clf.predict(X)

    def name(self):
        return self._name

class MostFrequent(Classifier):

    def __init__(self):
        self._name = 'most_frequent'

    def train(self, X, Y):
        self._mode = int(Y.mode())

    def predict(self, X):
        return pd.Series([self._mode]*len(X))

class DecisionTree(Classifier):

	def __init__(self):
		self._tree = DecisionTreeClassifier()
		self._name = 'decision_tree'

	def train(self, X, Y):
		self._tree.fit(X, Y)

	def predict(self, X):
		return self._tree.predict(X)