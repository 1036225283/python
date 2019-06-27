# encoding: utf-8

import tensorflow as tf

session = tf.Session()

saver = tf.train.Saver()

# module_file = saver.last_checkpoints("/Users/xws/Desktop/model/python/test.ckpt")
#
# saver.restore(session, module_file)
session.close()
