import os
os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from mock_data import regs_fixtures as rf
import tensorflow as tf
import numpy as np

regs_df = rf.load_fixture()

num_event = regs_df.event.nunique()
num_user = regs_df.user.nunique()
num_hidden_1 = 10
num_hidden_2 = 5

auto_encoder_value = tf.placeholder(tf.float64, [None, num_user])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_user, num_hidden_1], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_user], dtype=tf.float64)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_user], dtype=tf.float64)),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    return tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    return tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))


def train_model(epochs=20, batch_size=5):
    matrix = regs_df.pivot(index='event', columns='user', values='score')
    matrix.fillna(0, inplace=True)
    loss = tf.losses.mean_squared_error(auto_encoder_value, decoder(encoder(auto_encoder_value)))
    optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)
    matrix = matrix.as_matrix()
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        session.run(local_init)

        num_batches = int(matrix.shape[0] / batch_size)
        matrix = np.array_split(matrix, num_batches)

        for i in range(epochs):
            avg_cost = 0

            for batch in matrix:
                _, l = session.run([optimizer, loss], feed_dict={auto_encoder_value: batch})
                avg_cost += l

            avg_cost /= num_batches

            print("Training: {} Loss: {}".format(i + 1, avg_cost))
