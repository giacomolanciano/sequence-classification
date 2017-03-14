import functools
import tensorflow as tf
import matplotlib.pyplot as plt
from machine_learning.classifier_input import ClassifierInput
import numpy as np

EPOCH_NUM = 50


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
    def __init__(self, data, target, dropout, num_hidden=500, num_layers=10):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        # needed to initialize lazy properties
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.contrib.rnn.BasicLSTMCell(self._num_hidden)
        network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=self.dropout)
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        # Softmax layer.
        weight, bias = self._weight_and_bias(self._num_hidden, int(self.target.get_shape()[1]))

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        #cross_entropy = slim.losses.mean_squared_error(self.prediction, self.target)
        #cross_entropy = slim.losses.hinge_loss(self.prediction, self.target)
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.01
        # optimizer = tf.train.RMSPropOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
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


def main(considered_labels, input_size):
    # retrieve input data from database
    ml_input = ClassifierInput(input_size=input_size)
    ml_input.set_train_test_data(considered_labels)

    # create label-to-vector translation
    labels_vectors = []
    num_labels = len(considered_labels)
    for i in range(num_labels):
        label_vector = [0] * num_labels
        label_vector[i] = 1
        labels_vectors.append(label_vector)

    x_train = _format_data_matrix(ml_input.train_data)
    y_train = np.asarray([labels_vectors[i] for i in ml_input.train_labels])

    x_test = _format_data_matrix(ml_input.test_data)
    y_test = np.asarray([labels_vectors[i] for i in ml_input.test_labels])

    # initialize tensorflow
    _, rows, row_size = x_train.shape
    data = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.float32, [None, num_labels])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout)

    # start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_size = len(x_train)
    indices_num = int(train_size - (0.15 * train_size))
    err = []
    for epoch in range(EPOCH_NUM):
        rand_index = np.random.choice(train_size, indices_num)
        batch_xs = np.asarray(x_train[rand_index])
        batch_ys = y_train[rand_index]
        sess.run(model.optimize, {data: batch_xs, target: batch_ys, dropout: 0.4})

        # compute step error
        error = sess.run(model.error, {data: x_test, target: y_test, dropout: 0.4})
        error_percentage = 100*error
        err.append(error)
        print('Epoch {:2d} \n\taccuracy {:3.1f}% \n\terror {:3.1f}%'
              .format(epoch + 1, 100 - error_percentage, error_percentage))
    plt.figure(1)
    plt.plot([x for x in range(1, EPOCH_NUM+1)], err)
    plt.axis([1, 10, 0, 1.2])
    plt.show()

def _format_data_matrix(data):
    """
    Give the data matrix the right shape for being given as input to recurrent NN.
    :param data: a list of input data.
    :return: the formatted data matrix
    """
    data_matrix = np.asarray([data])
    transformed_data_matrix = []
    for seq in range(0, len(data_matrix[0])):
        row = data_matrix[0][seq]
        row = np.asarray([row])
        transformed_data_matrix.append(row)
    return np.asarray(transformed_data_matrix)


if __name__ == '__main__':
    main(['TRANSCRIPTION', 'LYASE', 'SIGNALING PROTEIN'], 1000)
