import functools
import sets
import tensorflow as tf
import numpy as np
from machine_learning.machine_learning_input import *

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

    def __init__(self, data, target, dropout, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.contrib.rnn.GRUCell(self._num_hidden)
        print(network)
        network = tf.contrib.rnn.DropoutWrapper(
            network, output_keep_prob=self.dropout)
        print(network)
        network = tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)
        print(network)
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        print(output)
        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        print(output)
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        print(last)
        print(weight)
        print(bias)
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        print('cross_entropy %s' % cross_entropy)
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        print('optimizer %s' % optimizer)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        print('mistakes %s' % mistakes)
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        print('weight %s' % weight)
        bias = tf.constant(0.1, shape=[out_size])
        print('bias %s' % bias)
        return tf.Variable(weight), tf.Variable(bias)


def main():
    # We treat images as sequences of pixel rows.

    my_ml_input = MachineLearningInput()
    my_ml_input.set_train_test_data(['TRANSCRIPTION', 'LYASE', 'SIGNALING PROTEIN'])

    n_attr = my_ml_input.max_feature_size

    X_train = np.asarray([my_ml_input.train_data])
    print(X_train.shape)
    X_train_mod = []
    for seq in range(0,len(X_train[0])):
        row = X_train[0][seq]
        row = np.asarray([row])
        X_train_mod.append(row)
    X_train = np.asarray(X_train_mod)
    print(X_train.shape)
    y_train = np.asarray(my_ml_input.train_labels)
    y_train_new = []

    for i in y_train:
        if(i==0):
            y_train_new.append([1,0,0])
        elif(i==1):
            y_train_new.append([0,1,0])
        else:
            y_train_new.append([0,0,1])

    y_train = np.asarray(y_train_new)


    X_test = np.asarray([my_ml_input.test_data])
    print(X_test.shape)

    X_test_mod = []
    for seq in range(0, len(X_test[0])):
        row = X_test[0][seq]
        row = np.asarray([row])
        X_test_mod.append(row)
    X_test = np.asarray(X_test_mod)
    print(X_test.shape)
    y_test = np.asarray(my_ml_input.test_labels)
    y_test_new = []
    for i in y_test:
        if (i == 0):
            y_test_new.append([1, 0, 0])
        elif (i == 1):
            y_test_new.append([0, 1, 0])
        else:
            y_test_new.append([0, 0, 1])

    y_test = np.asarray(y_test_new)

    _, rows, row_size = X_train.shape

    data = tf.placeholder(tf.float32, [None, rows, row_size])
    print(data)
    target = tf.placeholder(tf.float32, [None, 3])
    print(target)
    dropout = tf.placeholder(tf.float32)
    print(dropout)
    model = SequenceClassification(data, target, dropout)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for _ in range(100):
            rand_index = np.random.choice(len(X_train), 200)
            batch_xs = np.asarray(X_train[rand_index])
            batch_ys = y_train[rand_index]
            sess.run(model.optimize, {
                data: batch_xs, target: batch_ys, dropout: 1})
        print(X_test.shape)
        print(y_test.shape)
        error = sess.run(model.error, {
            data: X_test, target: y_test, dropout: 0.5})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))


if __name__ == '__main__':
    main()