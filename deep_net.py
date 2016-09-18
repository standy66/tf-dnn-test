"""A simple deep neural network."""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('Mnist_data', one_hot=True)

num_input = 784
hidden_size = 784
hidden_size_2 = 200
num_classes = 10
num_epochs = 30
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def model(x):
    """Define a model."""
    hl1 = {"weights": tf.Variable(tf.truncated_normal([num_input, hidden_size], stddev=0.5)),
           "biases": tf.Variable(tf.truncated_normal([hidden_size], stddev=0.5))}

    hl2 = {"weights": tf.Variable(tf.truncated_normal([hidden_size, hidden_size], stddev=0.5)),
           "biases": tf.Variable(tf.truncated_normal([hidden_size], stddev=0.5))}

    # l2l4 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size_2], stddev=0.5))

    # hl3 = {"weights": tf.Variable(tf.truncated_normal([hidden_size, hidden_size_2],
    #                                                   stddev=0.5)),
    #       "biases": tf.Variable(tf.truncated_normal([hidden_size_2], stddev=0.5))}

    # hl4 = {"weights": tf.Variable(tf.truncated_normal([hidden_size_2, hidden_size_2],
    #                                                   stddev=0.5)),
    #       "biases": tf.Variable(tf.truncated_normal([hidden_size_2], stddev=0.5))}

    output = {"weights": tf.Variable(tf.truncated_normal([hidden_size, num_classes], stddev=0.5)),
              "biases": tf.Variable(tf.truncated_normal([num_classes], stddev=0.5))}

    l1 = tf.matmul(x, hl1["weights"]) + hl1["biases"]
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1, hl2["weights"]) + hl2["biases"]
    l2 = tf.nn.relu(l2 + x)

    # l3 = tf.matmul(l2, hl3["weights"]) + hl3["biases"]
    # l3 = tf.nn.relu(l3)

    # l4 = tf.matmul(l3, hl4["weights"]) + hl4["biases"]
    # l4 = tf.nn.relu(l4 + tf.matmul(l2, l2l4))

    return tf.matmul(l2, output["weights"]) + output["biases"]

predicted = model(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predicted, y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(num_epochs):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, loss = sess.run([optimizer, cost], feed_dict={x: epoch_x,
                                                             y: epoch_y})
            epoch_loss += loss
        print("Epoch", epoch, "out of", num_epochs, "loss:", epoch_loss)

    correct = tf.equal(tf.argmax(predicted, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
