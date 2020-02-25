'''
convolutional neural network
tensorflow 
cross entropy loss function
relu convolution activation function
max pooling
softmax fully connected activation function
adam optimizer
'''

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import data_loader

def convolution_layer(input, channels, filters, filter_size=5, strides=1):
	weights = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, channels, filters], mean=0, stddev=0.1))
	biases = tf.Variable(tf.zeros([filters]))
	layer = tf.nn.conv2d(input, filter=weights, strides=[1, strides, strides, 1], padding='VALID') + biases
	return tf.nn.relu(layer)

def pooling_layer(input, k=2):
	return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	return tf.reshape(layer, [-1, num_features])

def fully_connected_layer(input, num_inputs, num_outputs, relu=True):
	weights = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], mean=0, stddev=0.1))
	biases = tf.Variable(tf.zeros([num_outputs]))
	layer = tf.matmul(input, weights) + biases
	if relu:
		return tf.nn.relu(layer)
	else:
		return layer


class NeuralNetwork:
	def __init__(self):
		self.sess = tf.Session()
		
		self.learning_rate = 0.001
		self.batch_size = 128
		
		self.load_data()
		self.build()
	
	def load_data(self):
		self.X_train, self.y_train = data_loader.load_training_data()
		self.X_valid, self.y_valid = data_loader.load_validation_data()
		self.X_test, self.y_test = data_loader.load_testing_data()
		
		self.img_shape = list(self.X_train[0].shape)
		self.num_classes = len(np.unique(self.y_train))
	
	def build(self):
		self.x = tf.placeholder(tf.float32, [None] + self.img_shape)
		self.y = tf.placeholder(tf.int32, [None])
		
		# Convolution 1: 32x32x3 -> 28x28x6 + ReLU
		conv1_channels = 3
		conv1_filters = 6
		conv1 = convolution_layer(self.x, conv1_channels, conv1_filters)
		# Pooling: 28x28x6 -> 14x14x6
		pool1 = pooling_layer(conv1)
		# Convolution 2: 14x14x6 -> 10x10x16 + ReLU
		conv2_channels = 6
		conv2_filters = 16
		conv2 = convolution_layer(pool1, conv2_channels, conv2_filters)
		# Pooling: 10x10x16 -> 5x5x16
		pool2 = pooling_layer(conv2)
		# Flatten: 5x5x16 -> 400
		flat = flatten_layer(pool2)
		# Fully Connected 1: 400 -> 120
		fc1_input = 400
		fc1_output = 120
		fc1 = fully_connected_layer(flat, fc1_input, fc1_output)
		# Fully Connected 2: 120 -> 84
		fc2_input = 120
		fc2_output = 84
		fc2 = fully_connected_layer(fc1, fc2_input, fc2_output)
		# Logits: 84 -> 43
		logits_input = 84
		logits_output = 43
		logits = fully_connected_layer(fc2, logits_input, logits_output, relu=False)
		
		one_hot_y = tf.one_hot(self.y, self.num_classes)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
		self.loss = tf.reduce_mean(cross_entropy)
		
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
		correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(one_hot_y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.prediction = tf.argmax(logits, axis=1)
	
	def train(self, epochs = 100):
		self.sess.run(tf.global_variables_initializer())
		
		print('training start')
		
		for i in range(epochs):
			X_data, y_data = shuffle(self.X_train, self.y_train)
			total_acc = 0
			for offset in range(0, len(y_data), self.batch_size):
				end = offset + self.batch_size
				batch_x, batch_y = X_data[offset:end], y_data[offset:end]
				feed_dict = {self.x: batch_x, self.y: batch_y}
				self.sess.run(self.optimizer, feed_dict=feed_dict)
				loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
				total_acc += acc * len(batch_y)
			
			feed_dict = {self.x: self.X_valid, self.y: self.y_valid}
			loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
			print(f'epoch {i + 1}: loss = {loss:.4f}, training accuracy = {total_acc / len(y_data):.4f}, validation accuracy = {acc:.4f}')
			
		print('training complete')
		
		feed_dict = {self.x: self.X_test, self.y: self.y_test}
		acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
		print(f'test accuracy = {acc:.4f}')
	
	def predict(self, x):
		feed_dict = {self.x : x}
		return self.sess.run(tf.argmax(self.prediction, axis=1), feed_dict=feed_dict)

if __name__ == '__main__':
	net = NeuralNetwork()
	net.train(20)