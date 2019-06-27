# encoding: utf-8

import tensorflow as tf

session = tf.Session()

# 创建变量，其中变量的shape和value都需要自己指定
A = tf.Variable(3, name="A")
B = tf.Variable(tf.zeros([3, 3]), name="B")

# 全局变量初始化  耗内存
# init = tf.global_variables_initializer()
# session.run(init)

# 单个变量初始化
session.run(A.initializer)
session.run(B.initializer)
print("create variable 3 = \n", session.run(A))
print("create variable 3*3 = \n", session.run(B))

# 变量的保存
saver = tf.train.Saver()
saver.save(session, "/Users/xws/Desktop/model/python/test.ckpt")
