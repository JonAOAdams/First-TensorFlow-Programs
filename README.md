# First-TensorFlow-Programs
It begins...
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
In [18]:
import tensorflow as tf
In [19]:
x = tf.placeholder(tf.float32, [None, 784])
In [20]:
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
In [21]:
y = tf.nn.softmax(tf.matmul(x, W) + b)
In [22]:
y_ = tf.placeholder(tf.float32, [None, 10])
In [23]:
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
In [24]:
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
In [25]:
sess = tf.InteractiveSession()
In [26]:
tf.global_variables_initializer().run()
In [27]:
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
In [28]:
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
In [29]:
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
In [30]:
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
0.9193
