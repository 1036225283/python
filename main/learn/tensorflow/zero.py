# encoding: utf-8

import tensorflow as tf

session = tf.Session()

# 初始化一个对象，其数据置为0
data = tf.zeros([2, 8], dtype=tf.float16)
print("创建2*8全为0的数据\n", session.run(data))

m2 = tf.constant([4, 4])
print("before zero copy = \n", session.run(m2))

# 拷贝一个对象，其数据置为0
data = tf.zeros_like(m2)
print("拷贝m2,并将数据置为0 = \n", session.run(data))

# 创建一个全为1的对象
data = tf.ones([2, 8], dtype=tf.int32)
print("创建2*8全为1的数据\n", session.run(data))

# 拷贝一个对象，其数据置为1
data = tf.ones_like(m2)
print("拷贝m2，并将数据置为1\n", session.run(data))

data = tf.fill([3, 3], 4)
print("创建一个对象，并填充指定数据\n", session.run(data))
