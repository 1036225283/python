# encoding: utf-8

import tensorflow as tf

# 计算图分为两部分

# 图的构建
m1 = tf.constant([3, 5])

# 图的执行
session = tf.Session()

# 对常量的初始化
m2 = tf.constant([1, 4])

# 初始化为0
m3 = tf.zeros([2, 2], tf.float32)
print("m3图结构 = ", m3)
print("m3图执行 = ", session.run(m3))

# 随机初始化
# tf.random_normal()、tf.truncated_normal()、tf.random_uniform()、tf.random_shuffle()


result = tf.add(m1, m2)

print("result图结构 = ", result)

print("result图执行 = ", session.run(result))

# 变量创建
tf.V
# 初始化、保存、加载


session.close()
