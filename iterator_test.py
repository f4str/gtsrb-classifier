import tensorflow as tf
import numpy as np

x_data = np.arange(1000, 1500)
y_data = np.arange(0, 500)
data = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(10).batch(5)

iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
train_initializer = iterator.make_initializer(data)
x, y = iterator.get_next()

with tf.Session() as sess:
	sess.run(train_initializer)
	try: 
		for i in range(100):
			batch_x, batch_y = sess.run([x, y])
			print(f'{i} : {batch_x}, {batch_y}')
	except tf.errors.OutOfRangeError:
		print('done')
		pass

