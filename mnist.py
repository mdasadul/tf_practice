import input_data
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides =[1,1,1,1],padding ='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides=[1,2,2,1],padding ='SAME')



mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,784])

W=tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# prediction
y = tf.nn.softmax(tf.matmul(x,W)+b)

# correct answer
y_ = tf.placeholder(tf.float32,[None,10])

cross_enropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_enropy)

init = tf.initialize_all_variables()


sess.run(init)
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))