# encoding: utf-8

import tensorflow as tf

session = tf.Session()

A = tf.Variable(3, name="A")
B = tf.Variable(tf.fill([3, 3], 5), name="B")
# session.run(init)
saver = tf.train.Saver()
# module_file = saver.last_checkpoints("/Users/xws/Desktop/model/python/test.ckpt")
#
saver.restore(session, "/Users/xws/Desktop/model/python/test.ckpt")
print(session.run(A))
print(session.run(B))
session.close()
