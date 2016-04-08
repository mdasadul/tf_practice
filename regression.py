import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data *.1 + 0.3

# predicting value for W and b that compute y_data = W * x_data +b

W = tf.Variable(tf.random_uniform([1],-1,1))
b = tf.Variable(tf.zeros([1]))

y = W *x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


#initialize the variable
init = tf.initialize_all_variables()

#lucnch the graph
sess = tf.Session()
sess.run(init)

for step in xrange(201):
	sess.run(train)
 	if(step%20 ==0):
 		print(step,sess.run(W),sess.run(b))
