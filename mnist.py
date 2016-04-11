import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

x = tf.placeholder(tf.float32,[None,784])

W=tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# prediction
y = tf.nn.softmax(tf.matmul(x,W)+b)

# correct answer
y_ = tf.placeholder(tf.float32,[None,10])

cross_enropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_enropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for i in range(10000):
    batch_xs,batch_ys = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))