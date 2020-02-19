import pickle
import numpy as np

def load_training_data(file='./data/train.p'):
	with open(file, mode='rb') as f:
		data = pickle.load(f)
	X_train, y_train = data['features'], data['labels']
	return X_train, y_train

def load_validation_data(file='./data/valid.p'):	
	with open(file, mode='rb') as f:
		data = pickle.load(f)
	X_valid, y_valid = data['features'], data['labels']
	return X_valid, y_valid
	
def load_testing_data(file='./data/test.p'):
	with open(file, mode='rb') as f:
		data = pickle.load(f)
	X_test, y_test = data['features'], data['labels']
	return X_test, y_test
