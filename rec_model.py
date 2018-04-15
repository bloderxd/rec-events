import os

os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from mock_data import regs_fixtures as rf
import tensorflow as tf
import numpy as np
import pandas as pd

regs_df = rf.load_fixture()
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


decoder_op = decoder(encoder(auto_encoder_value))


def train_model(data_frame, epochs=20, batch_size=5):
    matrix = data_frame.pivot(index='event', columns='user', values='score')
    matrix.fillna(0, inplace=True)
    events = matrix.index.tolist()
    users = matrix.columns.tolist()
    loss = tf.losses.mean_squared_error(auto_encoder_value, decoder_op)
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
    return matrix, events, users


def predict(matrix, events, users):
    matrix = np.concatenate(matrix, axis=0)
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    with tf.Session() as session:
        print('Predicting...')
        session.run(init)
        session.run(local_init)
        pre_preds = session.run(decoder_op, feed_dict={auto_encoder_value: matrix})
        predictions = pd.DataFrame()
        predictions = predictions.append(pd.DataFrame(pre_preds))
        predictions = predictions.stack().reset_index(name='score')
        predictions.columns = ['event', 'user', 'score']
        predictions['user'] = predictions['user'].map(lambda user: users[user])
        predictions['event'] = predictions['event'].map(lambda event: events[event])
    return predictions


def build_predictions():
    matrix, events, users = train_model(regs_df)
    return predict(matrix, events, users), regs_df
