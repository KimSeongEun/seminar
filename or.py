import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 2])
w = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1, 1]))
y = tf.sigmoid(tf.matmul(x, w) + b)
t = tf.placeholder(tf.float32, [None, 1])
l = tf.reduce_sum(tf.pow(tf.subtract(y, t), 2))

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [1]]

train = tf.train.GradientDescentOptimizer(0.1).minimize(l)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(10000):
    sess.run(train, feed_dict = {x: inputs, t: outputs})
    if step % 1000 == 0:
        print("step : " + str(step))
        print("loss = " + str(sess.run(l, feed_dict = {x: inputs, t: outputs})))
        for i in inputs:
            print("x = " + str(i) + " : y = " + str(sess.run(y, feed_dict = {x: [i]})))    
