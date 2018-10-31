import numpy as np
import pandas as pd
from math import sqrt

def accuracy(prediction, test_Y):
    survived = np.array(test_Y)
    absolute_accuracy = (survived==prediction).sum()
    accuracy = absolute_accuracy/len(test_Y)
    return 'accuracy', accuracy

def predicted1was1(prediction, test_Y):
    return 'predicted 1, was 1', (prediction.astype(bool) & test_Y.astype(bool)).sum()
    
def predicted1was0(prediction, test_Y):
    return 'predicted 1, was 0', (prediction.astype(bool) & ~test_Y.astype(bool)).sum()
    
def predicted0was1(prediction, test_Y):
    return 'predicted 0, was 1', (~prediction.astype(bool) & test_Y.astype(bool)).sum()
    
def predicted0was0(prediction, test_Y):
    return 'predicted 0, was 0', (~prediction.astype(bool) & ~test_Y.astype(bool)).sum()

evaluation_methods = [accuracy, predicted1was1, predicted0was0, predicted1was0, predicted0was1]

def evaluate(predictions, test_Y):
	results = []
	for pred in predictions:
		result = {' name' : pred[0]}
		for method in evaluation_methods:
			method_name, method_result = method(pred[1], test_Y)
			result[method_name] = method_result
		results.append(result)
	return pd.DataFrame(results)