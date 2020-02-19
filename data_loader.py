import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt

training_file = './data/train.p'
validation_file = './data/valid.p'
testing_file = './data/test.p'

with open(training_file, mode='rb') as f:
	train = pickle.load(f)
with open(validation_file, mode='rb') as f:
	valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
	test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

num_train = X_train.shape[0]
num_valid = X_valid.shape[0]
num_test = X_test.shape[0]

img_shape = X_train[0].shape
num_classes = len(np.unique(y_train))

print("Number of training examples: ", num_train)
print("Number of validation examples: ", num_valid)
print("Number of testing examples: ", num_test)
print("Image data shape =", img_shape)
print("Number of classes =", num_classes)

