# coding=utf-8
import tensorflow as tf

# 该图包含3个节点（两个源节点和乘法节点）
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
product = tf.matmul(matrix1, matrix2)

# 调用会话启动图
sess = tf.Session()
result = sess.run(product)

# 输出结果并关闭会话
print(result)
sess.close()

# 使用“with”代码块自动关闭, 该方法更简洁
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
