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
import data_loader

def convolution_layer(input, channels, filters, filter_size=5, strides=1):
	weights = tf.Variable(tf.random_normal([filter_size, filter_size, channels, filters]))
	biases = tf.Variable(tf.random_normal([filters]))
	layer = tf.nn.conv2d(input, filter=weights, strides=[1, strides, strides, 1], padding='SAME') + biases
	return tf.nn.relu(layer)

def pooling_layer(input, strides=1, k=2):
	return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	return tf.reshape(layer, [-1, num_features]), num_features

def fully_connected_layer(input, num_inputs, num_outputs, relu=True):
	weights = tf.Variable(tf.random_normal([num_inputs, num_outputs]))
	biases = tf.Variable(tf.random_normal([num_outputs]))
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
		X_train, y_train = data_loader.load_training_data()
		self.X_test, self.y_test = data_loader.load_testing_data()
		
		self.img_shape = list(X_train[0].shape)
		self.num_classes = len(np.unique(y_train))
		self.train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
	
	def build(self):
		self.x = tf.placeholder(tf.float32, [None] + self.img_shape)
		self.y = tf.placeholder(tf.int32, [None])
		self.one_hot_y = tf.one_hot(self.y, self.num_classes)
		
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
		flat, num_features = flatten_layer(pool2)
		# Fully Connected 1: 400 -> 120
		fc1_output = 120
		fc1 = fully_connected_layer(flat, num_features, fc1_output)
		# Fully Connected 2: 120 -> 84
		fc2_input = 120
		fc2_output = 84
		fc2 = fully_connected_layer(fc1, fc2_input, fc2_output)
		# Logits: 84 -> 43
		logits_input = 84
		logits_output = 43
		self.logits = fully_connected_layer(fc2, logits_input, logits_output, relu=False)
		
		self.prediction = tf.nn.softmax(self.logits)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)
		self.loss = tf.reduce_mean(cross_entropy)
		
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
		correct_prediction = tf.equal(tf.argmax(self.prediction, axis=1), tf.argmax(self.one_hot_y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	def train(self, epochs = 100):
		self.sess.run(tf.global_variables_initializer())
		
		for i in range(epochs):
			dataset = self.train_data.shuffle(10, reshuffle_each_iteration=True).batch(self.batch_size)
			iter = dataset.make_one_shot_iterator()
			x_batch, y_batch = iter.get_next()
			feed_dict = {self.x: x_batch.eval(session=self.sess), self.y: y_batch.eval(session=self.sess)}
			
			self.sess.run(self.optimizer, feed_dict=feed_dict)
			
			loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
			print(f'epoch {i + 1}: loss = {loss:.4f}, training accuracy = {acc:.4f}')
		print('training complete')
		
		feed_dict = {self.x: self.X_test, self.y: self.y_test}
		acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
		print(f'test accuracy = {acc:.4f}')
	
	def predict(self, x):
		feed_dict = {self.x : x}
		return self.sess.run(tf.argmax(self.prediction, axis=1), feed_dict=feed_dict)

if __name__ == '__main__':
	net = NeuralNetwork()
	net.train(500)
