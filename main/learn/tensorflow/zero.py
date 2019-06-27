# encoding: utf-8

import tensorflow as tf
from tensorflow import contrib as contrib
from tensorflow.python.util import *

from tensorflow import contrib as contrib

from tensorflow.python.util.lazy_loader import LazyLoader  # pylint: disable=g-import-not-at-top
# contrib = LazyLoader('contrib', globals(), 'tensorflow.contrib')
del LazyLoader



session = tf.Session()

contrib.__sizeof__()
tf.Session()
