import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import network1

net = network1.NeuralNetwork()

net.sess.run(tf.global_variables_initializer())
learning_rates = []
train_loss = []
train_acc = []
learning_rate = 1e-5

print('data start')
net.sess.run(net.train_initializer)
for i in range(100):
	learning_rate *= 1.1
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(net.loss)
	
	batch_x, batch_y = net.sess.run([net.X_batch, net.y_batch])
	feed_dict = {net.x: batch_x, net.y: batch_y}
	net.sess.run(optimizer, feed_dict=feed_dict)
	loss, acc = net.sess.run([net.loss, net.accuracy], feed_dict=feed_dict)
	
	learning_rates.append(learning_rate)
	train_loss.append(loss)
	train_acc.append(acc)
	
	print(f'iteration {i + 1}: learning rate = {learning_rate}')
print('data complete')

iterations = np.arange(len(learning_rates))

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title('Learning Rate vs Iteration')
ax1.set(xlabel='Iteration', ylabel='Learning Rate')
ax1.plot(iterations, learning_rates, 'b')

ax2.set_title('Loss vs Learning Rate')
ax2.set(xlabel='Learning Rate', ylabel='Loss')
ax2.plot(learning_rates, train_loss, 'b')

plt.show()
