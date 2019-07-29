import tensorflow as tf
import numpy as np

# 使用numpy生成100个随机点

x_data = np.random.rand(100)
y_data = x_data * 0.4 + 0.6

# 构造一个线性模型
b = tf.Variable(3.)
k = tf.Variable(2.)
y = k * x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))

# 定义一个梯度下降来继续训练的优化器

optimizer = tf.train.GradientDescentOptimizer(0.1)

# 最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for step in range(1001):
        session.run(train)
        if step % 20 == 0:
            print(step, session.run([k, b]))
