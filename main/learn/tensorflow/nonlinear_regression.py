import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个点
x_data = np.linspace(-0.5, 0.5, 100)[:, np.newaxis]  # -0.5 - 0.5的范围内生成200个点

noise = np.random.normal(0, 0.02, x_data.shape)
# print("x_data = ",x_data)

y_data = np.square(x_data) + noise

with tf.name_scope("input"):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 1], name="x-input")
    y = tf.placeholder(tf.float32, [None, 1], name="y-innput")

# 定义权值
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
WX_puls_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(WX_puls_b_L1)

# 定义输出层
weight_l2 = tf.Variable(tf.random_normal([10, 1]))
bias_l2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_l2 = tf.matmul(L1, weight_l2) + bias_l2

prediction = tf.nn.tanh(wx_plus_b_l2)

# 二次函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    #     session.run(train_step, feed_dict={x: x_data, y: y_data})
    ww = session.run(prediction, feed_dict={x: x_data, y: y_data})
    writer = tf.summary.FileWriter("/Users/xws/Desktop/data/mnist/", session.graph)
    #     print("ww = ",ww)
    for step in range(1000):
        session.run(train_step, feed_dict={x: x_data, y: y_data})
        if step % 50 == 0:
            # 获得预测值
            prediction_value = session.run(prediction, feed_dict={x: x_data})
            #     画图
            plt.figure()  # 新开一个绘图，都在绘图会叠加在一起
            plt.scatter(x_data, y_data)
            plt.plot(x_data, prediction_value)
            plt.draw()
            plt.pause(1)
