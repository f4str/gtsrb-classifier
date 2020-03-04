import pickle
import numpy as np

def preprocess(data, normalize):
	data = data.astype(np.float32)
	if normalize:
		for idx, img in enumerate(data):
			data[idx] = (img - np.mean(img)) / np.std(img)
		return data
	else:
		return data.astype(np.float32) / np.max(data)

def load_training_data(file='./data/train.p', normalize=True):
	with open(file, mode='rb') as f:
		data = pickle.load(f)
	X_train, y_train = data['features'], data['labels']
	X_train = preprocess(X_train, normalize)
	return X_train, y_train

def load_validation_data(file='./data/valid.p', normalize=True):	
	with open(file, mode='rb') as f:
		data = pickle.load(f)
	X_valid, y_valid = data['features'], data['labels']
	X_valid = preprocess(X_valid, normalize)
	return X_valid, y_valid
	
def load_testing_data(file='./data/test.p', normalize=True):
	with open(file, mode='rb') as f:
		data = pickle.load(f)
	X_test, y_test = data['features'], data['labels']
	X_test = preprocess(X_test, normalize)
	return X_test, y_test

if __name__ == "__main__":
	x, y = load_testing_data(normalize=True)
	print(f'x.shape = {x.shape}, x.dtype = {x.dtype}')
	print(f'x[0].shape = {x[0].shape}, x[0].dtype = {x[0].dtype}')
	print(f'x[0,0].shape = {x[0,0].shape}, x[0,0].dtype = {x[0,0].dtype}')
	print(f'x[0,0,0].shape = {x[0,0,0].shape}, x[0,0,0].dtype = {x[0,0,0].dtype}')
	print(f'x[0,0,0] = {x[0,0,0]}')
