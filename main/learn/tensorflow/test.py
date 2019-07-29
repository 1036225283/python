import tensorflow as tf

t = tf.constant([1, 2])
print(t)

# print(session)
# session.close()


x = tf.Variable([1, 1])
print("x = ", x)

sub = tf.subtract(t, x)
print("sub = ", sub)

state = tf.Variable(0, name="state")
new_val = tf.add(state, 1)

# 赋值op
update = tf.assign(state, new_val)

# Feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

mul = tf.multiply(input1, input2)
# 全局初始化
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    print("t = ", session.run(t))
    print("fetch ...")
    print(session.run([sub, state, new_val, update]))

    print("feed 的数据用字典给定")
    print(session.run(mul, feed_dict={input1: [3], input2: [2]}))
    for _ in range(5):
        session.run(update)
        print(session.run(state))
