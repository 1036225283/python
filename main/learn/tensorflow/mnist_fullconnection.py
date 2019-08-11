# mnist
# 使用简单的全连接神经网络
#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/Users/xws/Desktop/data/mnist/", one_hot=True)

batch_size = 500
n_batch = mnist.train.num_examples
print(mnist.train.num_examples)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建简单的神经网络
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, w) + b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 正确率
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as s:
    s.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            print("batch = " + str(batch))
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            s.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
            acc = s.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("epoch = " + str(epoch) + " acc = " + str(acc))
        acc = s.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("epoch = " + str(epoch) + " acc = " + str(acc))
