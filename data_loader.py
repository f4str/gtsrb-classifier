import numpy as np
import pandas as pd
from PIL import Image


def load_train(directory, size=(32, 32)):
	df = pd.read_csv(directory + '/Train.csv')
	paths = df['Path'].tolist()
	
	images = []
	for path in paths:
		image = Image.open(f'{directory}/{path}')
		image = image.resize(size)
		images.append(np.array(image))
	
	X_train = np.array(images, dtype=np.float32) / 255
	y_train = df['ClassId'].to_numpy()
	
	return X_train, y_train


def load_test(directory, size=(32, 32)):
	df = pd.read_csv(directory + '/Test.csv')
	paths = df['Path'].tolist()
	
	images = []
	for path in paths:
		image = Image.open(f'{directory}/{path}')
		image = image.resize(size)
		images.append(np.array(image))
	
	X_test = np.array(images, dtype=np.float32) / 255
	y_test = df['ClassId'].to_numpy()
	
	return X_test, y_test


if __name__ == "__main__":
	X_train, y_train = load_train('data')
	X_test, y_test = load_test('data')
	
	print(f'X_train shape: {X_train.shape}, X_train dtype: {X_train.dtype}')
	print(f'y_train shape: {y_train.shape}, y_train dtype: {y_train.dtype}')
	print(f'X_test shape: {X_test.shape}, X_test dtype: {X_test.dtype}')
	print(f'y_test shape: {y_test.shape}, y_test dtype: {y_test.dtype}')
	print(f'number of class {len(np.unique(y_train))}')
