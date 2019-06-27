# encoding: utf-8

import tensorflow as tf

session = tf.Session()

a = tf.placeholder(shape=[2], dtype=tf.float32)
b = tf.constant(tf.fill([3, 3], 3), dtype=tf.float32)

c = tf.add(a, b)
print(session.run(c))

session.close()
