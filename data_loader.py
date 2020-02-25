import pickle
import numpy as np

def load_training_data(file='./data/train.p'):
	with open(file, mode='rb') as f:
		data = pickle.load(f)
	X_train, y_train = data['features'], data['labels']
	X_train = (255 - X_train.astype(np.float32)) / 255
	return X_train, y_train

def load_validation_data(file='./data/valid.p'):	
	with open(file, mode='rb') as f:
		data = pickle.load(f)
	X_valid, y_valid = data['features'], data['labels']
	X_valid = (255 - X_valid.astype(np.float32)) / 255
	return X_valid, y_valid
	
def load_testing_data(file='./data/test.p'):
	with open(file, mode='rb') as f:
		data = pickle.load(f)
	X_test, y_test = data['features'], data['labels']
	X_test = (255 - X_test.astype(np.float32)) / 255
	return X_test, y_test

if __name__ == "__main__":
	x, y = load_testing_data()
	print(f'x.shape = {x.shape}, x.dtype = {x.dtype}')
	print(f'x[0].shape = {x[0].shape}, x[0].dtype = {x[0].dtype}')
	print(f'x[0,0].shape = {x[0,0].shape}, x[0,0].dtype = {x[0,0].dtype}')
	print(f'x[0,0,0].shape = {x[0,0,0].shape}, x[0,0,0].dtype = {x[0,0,0].dtype}')
	print(f'x[0,0,0] = {x[0,0,0]}')
