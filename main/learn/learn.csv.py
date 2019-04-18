import tensorflow as tf
import tempfile

tf.enable_eager_execution

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file

_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)
