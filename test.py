import tensorflow as tf
import numpy as np

a=np.array([1,2])
b=np.array([0,1])
c=np.array([3,4])

x=tf.placeholder(dtype=tf.float32,shape=(None))
y=tf.placeholder(dtype=tf.float32,shape=(None))
z=tf.placeholder(dtype=tf.float32,shape=(None))

e = tf.map_fn(lambda x: tf.add(x[0],tf.multiply(tf.subtract(tf.constant(1,tf.float32),x[1]),tf.multiply(tf.constant(.99,tf.float32),x[2]))), (x,y,z), dtype=tf.float32)


with tf.Session() as sess:
	print sess.run(e,feed_dict={x:a,y:b,z:c})
