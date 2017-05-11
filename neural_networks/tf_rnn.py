import os
import functools
from datetime import timedelta

import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np

from machine_learning.sequence_classifier_input import SequenceClassifierInput
from utils.constants import TRAINED_MODELS_FOLDER, TF_MODEL_EXT

INPUTS_PER_LABEL = 100
NEURONS_NUM = 200
LAYERS_NUM = 3
LEARNING_RATE = 0.003
EPOCHS_NUM = 10
STEPS_NUM = 100
MINI_BATCH_SIZE = 0.3
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
    def __init__(self, data, target, dropout_keep_prob, neurons_num=NEURONS_NUM, layers_num=LAYERS_NUM):
        self.data = data
        self.target = target
        self.dropout_keep_prob = dropout_keep_prob
        self._neurons_num = neurons_num
        self._layers_num = layers_num
        # needed to initialize lazy properties
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cell = tf.contrib.rnn.BasicLSTMCell(self._neurons_num)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        network = tf.contrib.rnn.MultiRNNCell([cell] * self._layers_num)

        # discard the state, since every time we look at a new sequence it becomes irrelevant.
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32, sequence_length=self.length)

        # Select last output.
        last = self._last_relevant(output, self.length)

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

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant


def main(considered_labels=None, cached_dataset=None, inputs_per_label=1000):
    # retrieve input data from database
    clf_input = SequenceClassifierInput(
        considered_labels=considered_labels,
        cached_dataset=cached_dataset,
        inputs_per_label=inputs_per_label
    )

    train_data, test_data, train_labels, test_labels = clf_input.get_rnn_train_test_data()

    """
    INITIALIZE COMPUTATION GRAPH
    """
    sequence_max_length = len(train_data[0])
    frame_dimension = len(train_data[0][0])

    # sequences number (i.e. batch_size) defined at runtime
    data = tf.placeholder(tf.float32, [None, sequence_max_length, frame_dimension])
    target = tf.placeholder(tf.float32, [None, clf_input.labels_num])
    dropout_keep_prob = tf.placeholder(tf.float32)
    model = SequenceClassifier(data, target, dropout_keep_prob)

    # to save and restore variables after training
    saver = tf.train.Saver()

    # start session
    start_time = time.time()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_size = len(train_data)
    indices_num = int(MINI_BATCH_SIZE * train_size)
    errors = []
    for epoch in range(EPOCHS_NUM):
        print('Epoch {:2d}'.format(epoch + 1))

        for step in range(STEPS_NUM):
            print('\tstep {:3d}'.format(step + 1))
            rand_index = np.random.choice(train_size, indices_num)
            mini_batch_xs = train_data[rand_index]
            mini_batch_ys = train_labels[rand_index]
            sess.run(model.optimize, {data: mini_batch_xs, target: mini_batch_ys, dropout_keep_prob: DROPOUT_KEEP_PROB})

        # dropout_keep_prob is set to 1 (i.e. keep all) only for testing
        error = sess.run(model.error, {data: test_data, target: test_labels, dropout_keep_prob: 1})
        error_percentage = 100 * error
        errors.append(error)
        print('\taccuracy: {:3.1f}% \n\terror: {:3.1f}%'.format(100 - error_percentage, error_percentage))

    elapsed_time = (time.time() - start_time)
    print('RNN running time:', timedelta(seconds=elapsed_time))

    # save model variables
    saver.save(sess, os.path.join(TRAINED_MODELS_FOLDER, str(time.time()) + TF_MODEL_EXT))

    """
    PLOT ERROR FUNCTION
    """
    plt.figure(1)
    plt.plot([x for x in range(1, EPOCHS_NUM + 1)], errors)
    plt.axis([1, EPOCHS_NUM, 0, 1])
    plt.show()


if __name__ == '__main__':
    # main(considered_labels=['OXIDOREDUCTASE', 'PROTEIN TRANSPORT'], inputs_per_label=INPUTS_PER_LABEL)
    main(cached_dataset='1494155537.5589833_3_OXIDOREDUCTASE_PROTEIN TRANSPORT_.pickle')
