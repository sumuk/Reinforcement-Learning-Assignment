from __future__ import division
import pickle
import tensorflow as tf
import numpy as np
f=open('/home/sumuk/Desktop/sumuk/homework-master/hw1/data.pkl','rb')
data=pickle.load(f)
f.close()


action=data['actions']
action=action.reshape((action.shape[0],action.shape[-1]))
#action=np.interp(action,[np.amin(action),np.amax(action)],[-1,1])
observation=data['observations']

x=tf.placeholder(tf.float32, shape=[None, observation.shape[-1]],name="x")
y=tf.placeholder(tf.float32, shape=[None, action.shape[-1]],name="y")
layer={}
layer[0]=x
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def fc_layer(previous, input_size, output_size,i):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(previous, W,name=i) + b
	

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
def run_network(dims,batch_size,epoch):
	for i in range(len(dims)-1):
		if i==len(dims)-2:
			
			layer[i+1]=tf.nn.xw_plus_b(layer[i],weight_variable([dims[i],dims[i+1]]),bias_variable([dims[i+1]]),str(i))#fc_layer(layer[i],dims[i],dims[i+1],str(i))
		else:
			layer[i+1]=tf.nn.tanh(fc_layer(layer[i],dims[i],dims[i+1],str(i)),name=str(i))
	last_layer=layer[len(dims)-1]
	
        loss = tf.reduce_mean(tf.losses.absolute_difference(last_layer,y))
	train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
	#equality = np.isclose(layer[len(dims)-1],y)
	#acc = tf.reduce_mean(False if False in equality else True)
        saver = tf.train.Saver()
	
	tf.add_to_collection("last_layer", last_layer)
	#tf.add_to_collection("adam", last_layer)
	#train_step=tf.train.GradientDescentOptimizer(.07).minimize(loss)
	no_of_batch=int(observation.shape[0]/batch_size)
	last_batch=observation.shape[0]%batch_size
	
	with tf.Session() as sess:
           sess.run(tf.global_variables_initializer())
	   #saver.restore(sess,'/home/sumuk/Desktop/sumuk/my_test_model-1')
	   for i in range(epoch):
		#shuffle_in_unison(observation,action)
		for k in range(no_of_batch):
			obser=observation[k:k+batch_size]
			act=action[k:k+batch_size]
		if last_batch:
			obser=observation[no_of_batch*batch_size:]
			act=action[no_of_batch*batch_size:]
		
		sess.run(train_step,feed_dict={x :obser,y:act})
		if (i%500 ==0):
			output1=sess.run(last_layer,feed_dict={x :obser,y:act})
			num=[np.allclose(output1[i],act[i]) for i in range(obser.shape[0])].count(True)
			#print num/output1.shape[0]
			print "action",sess.run(last_layer,feed_dict={x :np.array([obser[0]]),y:np.array([act[0]])}),act[0]
			print "loss at",i,sess.run(loss,feed_dict={x :obser,y:act})
	   saver.save(sess, '/home/sumuk/Desktop/sumuk/my_test_model')
	   print "save the file"
dims=[observation.shape[-1],200,100,action.shape[-1]]
batch_size=256
epoch=50000

run_network(dims,batch_size,epoch)


		
