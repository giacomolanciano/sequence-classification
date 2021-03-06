import os
import sys

import functools
from datetime import timedelta

import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
from memory_profiler import profile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))
from machine_learning.sequence_classifier_input import SequenceClassifierInput
from utils.constants import TRAINED_MODELS_FOLDER, TF_MODEL_EXT, IMG_EXT, FILENAME_SEPARATOR
from utils.files import unique_filename

import inspect

CONSIDERED_LABELS = ['HYDROLASE', 'TRANSFERASE']
INPUTS_PER_LABEL = 1000
NEURONS_NUM = 100
LAYERS_NUM = 3
LEARNING_RATE = 0.003
EPOCHS_NUM = 10
STEPS_NUM = 100
MINI_BATCH_SIZE = 0.3
DROPOUT_KEEP_PROB = 0.5


def lazy_property(funct):
    """
    Causes the function to act like a property. The function is only evaluated once, when it is accessed for the
    first time. The result is stored and directly returned for later accesses, for the sake of efficiency.
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
        def lstm_cell():
            # With the latest TensorFlow source code (as of Mar 27, 2017),
            # the BasicLSTMCell will need a reuse parameter which is unfortunately not
            # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
            # an argument check here:
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(
                    self._neurons_num, forget_bias=0.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    self._neurons_num, forget_bias=0.0, state_is_tuple=True)

        # Recurrent network.
        attn_cell = lstm_cell
        if DROPOUT_KEEP_PROB < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.dropout_keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell()] * self._layers_num, state_is_tuple=True)

        # discard the state, since every time we look at a new sequence it becomes irrelevant.
        output, _ = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32, sequence_length=self.length)

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


@profile
def main(considered_labels=None, cached_dataset=None, inputs_per_label=1000, spectrum=3):
    # retrieve input data from database
    clf_input = SequenceClassifierInput(
        considered_labels=considered_labels,
        cached_dataset=cached_dataset,
        inputs_per_label=inputs_per_label,
        spectrum=spectrum
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

    print('Inputs per label:  {0}'.format(clf_input.inputs_per_label))
    print('Neurons per layer: {0}'.format(NEURONS_NUM))
    print('Dropout keep prob: {0}'.format(DROPOUT_KEEP_PROB))

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
    model_checkpoint_time = str(int(time.time()))
    model_checkpoint_dir = os.path.join(TRAINED_MODELS_FOLDER, model_checkpoint_time)
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir)
    saver.save(sess, os.path.join(model_checkpoint_dir, model_checkpoint_time) + TF_MODEL_EXT)

    """
    PLOT ERROR FUNCTION
    """
    _, fig_basename = unique_filename(os.path.join(model_checkpoint_dir, clf_input.dump_basename))
    fig = fig_basename + IMG_EXT
    fig_zoom = FILENAME_SEPARATOR.join([fig_basename, 'zoom']) + IMG_EXT
    fig_avg = FILENAME_SEPARATOR.join([fig_basename, 'avg']) + IMG_EXT

    measures_num = EPOCHS_NUM * STEPS_NUM
    plt.figure()
    plt.plot(range(1, measures_num + 1), errors)
    plt.axis([1, measures_num, 0, 1])
    plt.savefig(fig, bbox_inches='tight')

    plt.figure()
    plt.plot(range(1, measures_num + 1), errors)
    plt.savefig(fig_zoom, bbox_inches='tight')

    plt.figure()
    # group steps errors of the same epoch and compute the average error in epoch
    plt.plot(range(1, EPOCHS_NUM + 1), [sum(group) / STEPS_NUM for group in zip(*[iter(errors)]*STEPS_NUM)])
    plt.savefig(fig_avg, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    # main(considered_labels=CONSIDERED_LABELS, inputs_per_label=INPUTS_PER_LABEL, spectrum=3)
    main(cached_dataset='1494941406_3_1000_HYDROLASE_TRANSFERASE')
