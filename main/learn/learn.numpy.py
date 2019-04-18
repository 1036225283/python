import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

ndarray = np.ones([3, 3])

print(ndarray)

print("TensorFlow operations convert numpy arrays to Tensors automatically")
print("\n")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 3))

print("\n tf.add(tensor)")

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())


x = tf.random_uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))