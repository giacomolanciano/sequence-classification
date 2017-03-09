# Example for my blog post at:
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
import functools
import sets
import tensorflow as tf
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

    def __init__(self, data, target, dropout, num_hidden=200, num_layers=2):
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
        network = tf.contrib.rnn.DropoutWrapper(
            network, output_keep_prob=self.dropout)
        network = tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)
        print(network)
        print(self.data)
        #output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        output_old = my_ml_input.test_labels
        output = []
        for i in output_old:
            if (i == 0):
                output.append([1, 0, 0])
            elif (i == 1):
                output.append([0, 1, 0])
            else:
                output.append([0, 0, 1])
        output = np.asarray(output)
        # Select last output.
        output = tf.transpose(output)
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def main():
    # We treat images as sequences of pixel rows.
    my_ml_input = MachineLearningInput()
    my_ml_input.set_train_test_data(['TRANSCRIPTION', 'LYASE', 'SIGNALING PROTEIN'])

    n_attr = my_ml_input.max_feature_size

    X_train = np.asarray(my_ml_input.train_data)
    y_train = np.asarray(my_ml_input.train_labels)

    y_train_new = []

    for i in y_train:
        if (i == 0):
            y_train_new.append([1, 0, 0])
        elif (i == 1):
            y_train_new.append([0, 1, 0])
        else:
            y_train_new.append([0, 0, 1])
    y_train_new = np.asarray(y_train_new)

    X_test = my_ml_input.test_data
    X_test = np.asarray(X_test)
    y_test = my_ml_input.test_labels
    y_test_new = []
    for i in y_test:
        if (i == 0):
            y_test_new.append([1, 0, 0])
        elif (i == 1):
            y_test_new.append([0, 1, 0])
        else:
            y_test_new.append([0, 0, 1])
    y_test_new = np.asarray(y_test_new)

    _, rows, row_size = None,len(my_ml_input.train_data),n_attr
    num_classes = 3
    data = tf.placeholder(tf.float32, [None, row_size])
    print(data)
    target = tf.placeholder(tf.float32, [None, num_classes])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for _ in range(100):
            rand_index = np.random.choice(len(X_train), 100)
            batch_xs = X_train[rand_index]
            batch_ys = y_train_new[rand_index]
            sess.run(model.optimize, {
                data: batch_xs, target: batch_ys, dropout: 0.5})
        error = sess.run(model.error, {
            data: X_test, target: y_test_new, dropout: 1})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))


if __name__ == '__main__':
    main()