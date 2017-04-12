import functools
import tensorflow as tf
import matplotlib.pyplot as plt
from machine_learning.sequence_classifier import SequenceClassifierInput
import numpy as np
from neural_networks import tf_glove

EPOCH_NUM = 500
LEARNING_RATE = 0.003


def lazy_property(function):
    """
    Causes the function to act like a property. The function is only evaluated once, when it's accessed for the
    first time. The result is stored an directly returned for later accesses, for tha sake of efficiency.
    """
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class SequenceClassification:
    def __init__(self, data, target, dropout, num_neurons=200, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_neurons = num_neurons
        self._num_layers = num_layers
        # needed to initialize lazy properties
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.contrib.rnn.BasicLSTMCell(self._num_neurons)
        network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=self.dropout)
        network = tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)

        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        # Softmax layer.
        weight, bias = self._weight_and_bias(self._num_neurons, int(self.target.get_shape()[1]))

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        # cross_entropy = slim.losses.mean_squared_error(self.prediction, self.target)
        # cross_entropy = slim.losses.hinge_loss(self.prediction, self.target)
        return cross_entropy

    @lazy_property
    def optimize(self):
        # optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
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
    :return: the formatted data matrix
    """
    data_matrix = np.asarray([data])
    transformed_data_matrix = []
    for seq in range(0, len(data_matrix[0])):
        row = data_matrix[0][seq]
        row = np.asarray([row])
        transformed_data_matrix.append(row)
    return np.asarray(transformed_data_matrix)


def main(considered_labels, inputs_per_label):
    # retrieve input data from database
    clf_input = SequenceClassifierInput(considered_labels, inputs_per_label=inputs_per_label)

    # create label-to-vector translation structure
    labels_vectors = []
    num_labels = len(considered_labels)
    for i in range(num_labels):
        label_vector = [0] * num_labels
        label_vector[i] = 1
        labels_vectors.append(label_vector)

    train_data, test_data, train_labels, test_labels = clf_input.get_rnn_train_test_data()
    train_size = len(train_data)

    input_data = train_data + test_data
    glove_model = tf_glove.GloVeModel(embedding_size=100, context_size=10)
    glove_model.fit_to_corpus(input_data)
    glove_model.train(num_epochs=100)

    glove_matrix = []
    for shingle_list in input_data:
        vectors = []
        for shingle in shingle_list:
            vec = glove_model.embedding_for(shingle)
            vectors.append(vec)
        glove_matrix.append(np.mean(vectors, axis=0))
    glove_matrix = np.asarray(glove_matrix)

    train_data = glove_matrix[:train_size]
    print(train_data.shape)
    test_data = glove_matrix[train_size:]
    print(test_data.shape)

    train_data = _format_data_matrix(train_data)
    train_labels = np.asarray([labels_vectors[i] for i in train_labels])
    test_data = _format_data_matrix(test_data)
    test_labels = np.asarray([labels_vectors[i] for i in test_labels])

    # initialize tensorflow
    _, rows, row_size = train_data.shape

    data = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.float32, [None, num_labels])
    dropout = tf.placeholder(tf.float32)

    model = SequenceClassification(data, target, dropout)

    # start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_size = len(train_data)
    indices_num = int(train_size - (0.25 * train_size))
    err = []

    for epoch in range(EPOCH_NUM):
        rand_index = np.random.choice(train_size, indices_num)
        batch_xs = np.asarray(train_data[rand_index])
        batch_ys = train_labels[rand_index]
        sess.run(model.optimize, {data: batch_xs, target: batch_ys, dropout: 0.5})

        # compute step error
        error = sess.run(model.error, {data: test_data, target: test_labels, dropout: 1})
        error_percentage = 100 * error
        err.append(error)
        print('Epoch {:2d} \n\taccuracy {:3.1f}% \n\terror {:3.1f}%'
              .format(epoch + 1, 100 - error_percentage, error_percentage))

    # plot error function
    plt.figure(1)
    plt.plot([x for x in range(1, EPOCH_NUM + 1)], err)
    plt.axis([1, EPOCH_NUM, 0, 1])
    plt.show()


if __name__ == '__main__':
    main(['OXIDOREDUCTASE', 'PROTEIN TRANSPORT'], 10)
