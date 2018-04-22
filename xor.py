import tensorflow as tf

middle = 4

x1 = tf.placeholder(tf.float32, [None, 2])
w1 = tf.Variable(tf.random_normal([2, middle]))
b1 = tf.Variable(tf.random_normal([1, middle]))
y1 = tf.sigmoid(tf.matmul(x1, w1) + b1)

x2 = y1
w2 = tf.Variable(tf.random_normal([middle, 1]))
b2 = tf.Variable(tf.random_normal([middle, 1]))
y2 = tf.sigmoid(tf.matmul(x2, w2) + b2)

t = tf.placeholder(tf.float32, [None, 1])
l = tf.reduce_sum(tf.pow(tf.subtract(y2, t), 2))

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

train = tf.train.GradientDescentOptimizer(0.1).minimize(l)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(10000):
    sess.run(train, feed_dict = {x1: inputs, t: outputs})
    if step % 1000 == 0:
        print("step : " + str(step))
        print("loss = " + str(sess.run(l, feed_dict = {x1: inputs, t: outputs})))
        for i in inputs:
            print("x = " + str(i) + " : y = " + str(sess.run(y2, feed_dict = {x1: [i]})))    
