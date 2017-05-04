import functools
from datetime import timedelta

import tensorflow as tf
import matplotlib.pyplot as plt
import time

from machine_learning.sequence_classifier_input import SequenceClassifierInput
import numpy as np

LEARNING_RATE = 0.003
EPOCHS_NUM = 10
STEPS_NUM = 10
BATCH_SIZE = 0.1
DROPOUT_KEEP_PROB = 0.5


def lazy_property(funct):
    """
    Causes the function to act like a property. The function is only evaluated once, when it's accessed for the
    first time. The result is stored an directly returned for later accesses, for tha sake of efficiency.
    """
    attribute = '_' + funct.__name__

    @property
    @functools.wraps(funct)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, funct(self))
        return getattr(self, attribute)

    return wrapper


class SequenceClassifier:
    def __init__(self, data, target, dropout, neurons_num=200, layers_num=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._neurons_num = neurons_num
        self._layers_num = layers_num
        # needed to initialize lazy properties
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cell = tf.contrib.rnn.BasicLSTMCell(self._neurons_num)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
        network = tf.contrib.rnn.MultiRNNCell([cell] * self._layers_num)

        # discard the state, since every time we look at a new sequence it becomes irrelevant.
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        # Softmax layer.
        weight, bias = self._weight_and_bias(self._neurons_num, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def _format_data_matrix(data):
    """
    Give the data matrix the right shape for being given as input to recurrent NN.
    :param data: a list of input data.
    :return: the formatted data matrix.
    """
    data_matrix = np.asarray([data])
    transformed_data_matrix = []
    for seq in range(0, len(data_matrix[0])):
        row = data_matrix[0][seq]
        row = np.asarray([row])
        transformed_data_matrix.append(row)
    return np.asarray(transformed_data_matrix)


def main(considered_labels=None, cached_dataset=None, inputs_per_label=1000):
    # retrieve input data from database
    clf_input = SequenceClassifierInput(considered_labels=considered_labels, cached_dataset=cached_dataset,
                                        inputs_per_label=inputs_per_label)

    train_data, test_data, train_labels, test_labels = clf_input.get_rnn_train_test_data()
    labels_num = clf_input.labels_num

    """
    INITIALIZE TENSORFLOW COMPUTATIONAL GRAPH
    """
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    print(train_data.shape)
    _, rows, row_size = train_data.shape

    data = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.float32, [None, labels_num])
    dropout = tf.placeholder(tf.float32)

    model = SequenceClassifier(data, target, dropout)

    # start session
    start_time = time.time()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_size = len(train_data)
    indices_num = int(BATCH_SIZE * train_size)
    errors = []
    for epoch in range(EPOCHS_NUM):
        print('Epoch {:2d}'.format(epoch + 1))
        error = 0

        for step in range(STEPS_NUM):
            print('\tstep {:3d}'.format(step + 1))
            rand_index = np.random.choice(train_size, indices_num)
            batch_xs = train_data[rand_index]
            batch_ys = train_labels[rand_index]
            sess.run(model.optimize, {data: batch_xs, target: batch_ys, dropout: DROPOUT_KEEP_PROB})
            error += sess.run(model.error, {data: test_data, target: test_labels, dropout: 1})  # keep all for testing

        error = error / STEPS_NUM
        error_percentage = 100 * error
        errors.append(error)
        print('\taccuracy {:3.1f}% \n\terror {:3.1f}%'.format(100 - error_percentage, error_percentage))

    elapsed_time = (time.time() - start_time)
    print('RNN running time:', timedelta(seconds=elapsed_time))

    """
    PLOT ERROR FUNCTION
    """
    plt.figure(1)
    plt.plot([x for x in range(1, EPOCHS_NUM + 1)], errors)
    plt.axis([1, EPOCHS_NUM, 0, 1])
    plt.show()


if __name__ == '__main__':
    # main(considered_labels=['OXIDOREDUCTASE', 'PROTEIN TRANSPORT'], inputs_per_label=100)
    main(cached_dataset='1493894120.6477304_3_OXIDOREDUCTASE_PROTEIN TRANSPORT_.pickle')
