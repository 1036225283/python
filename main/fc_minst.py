import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

# 导入数据, 强烈建议预先下载
mnist = input_data.read_data_sets("/Users/xws/Desktop/data/", one_hot=True)

# 训练集占位符：28*28=784
x = tf.placeholder(tf.float32, [None, 784])
# 初始化参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 输出结果
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 真实值
y_ = tf.placeholder(tf.float32, [None, 10])
# 计算交叉熵
crossEntropy = -tf.reduce_sum(y_ * tf.log(y))
# 训练策略
trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropy)
# 初始化参数值
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 开始训练:循环训练1000次
for i in range(1000):
    batchXs, batchYs = mnist.train.next_batch(100)
    sess.run(trainStep, feed_dict={x: batchXs, y_: batchYs})

# 评估模型
correctPrediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
a =  sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

print(a)
print("end")
