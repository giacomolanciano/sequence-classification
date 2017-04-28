import os
from datetime import timedelta

import numpy as np
from sklearn.model_selection import train_test_split

from neural_networks import tf_glove
from utils import persistence
from utils.constants import \
    PADDING_VALUE, SPECTRUM_KEY, LABELS_KEY, INPUTS_PER_LABEL_KEY, TIME_KEY, DATASET_KEY, RNN_SUFFIX, SPECTRUM_SUFFIX, \
    FILENAME_SEPARATOR, DUMP_EXT, DATA_FOLDER

import pickle
import time

BASE_TWO = 2
AMINO_ACIDS_DICT = {
    'A': '10000000000000000000000000',
    'B': '01000000000000000000000000',
    'C': '00100000000000000000000000',
    'D': '00010000000000000000000000',
    'E': '00001000000000000000000000',
    'F': '00000100000000000000000000',
    'G': '00000010000000000000000000',
    'H': '00000001000000000000000000',
    'I': '00000000100000000000000000',
    'J': '00000000010000000000000000',  # never occurs
    'K': '00000000001000000000000000',
    'L': '00000000000100000000000000',
    'M': '00000000000010000000000000',
    'N': '00000000000001000000000000',
    'O': '00000000000000100000000000',
    'P': '00000000000000010000000000',
    'Q': '00000000000000001000000000',
    'R': '00000000000000000100000000',
    'S': '00000000000000000010000000',
    'T': '00000000000000000001000000',
    'U': '00000000000000000000100000',
    'V': '00000000000000000000010000',
    'W': '00000000000000000000001000',
    'X': '00000000000000000000000100',
    'Y': '00000000000000000000000010',
    'Z': '00000000000000000000000001'
}
GLOVE_EMBEDDING_SIZE = 100


class MissingInputError(Exception):
    pass


class SequenceClassifierInput(object):
    def __init__(self, considered_labels=None, cached_dataset=None, table_name='protein', inputs_per_label=1000,
                 spectrum=3, test_size=0.25, random_state=42, progress=True):
        self.progress = progress
        self.considered_labels = considered_labels
        self.cached_dataset = cached_dataset
        self.table_name = table_name
        self.inputs_per_label = inputs_per_label
        self.spectrum = spectrum
        self.test_size = test_size
        self.random_state = random_state
        self.time = None

        # initialize training and test splits (both data and labels)
        if considered_labels:
            self.cached_dataset = None
            self.labels_num = len(considered_labels)
            self.train_data, self.test_data, self.train_labels, self.test_labels = \
                self._get_training_inputs_by_labels(
                    considered_labels, table_name, inputs_per_label, test_size, random_state)
        elif cached_dataset:
            dataset_dict = self._load_dataset(cached_dataset)
            self.spectrum = dataset_dict[SPECTRUM_KEY]
            self.considered_labels = dataset_dict[LABELS_KEY]
            self.labels_num = len(self.considered_labels)
            self.inputs_per_label = dataset_dict[INPUTS_PER_LABEL_KEY]
            self.time = dataset_dict[TIME_KEY]
            self.train_data, self.test_data, self.train_labels, self.test_labels = dataset_dict[DATASET_KEY]
        else:
            raise MissingInputError('Neither labels to be considered nor cached dataset are provided.')

    def get_rnn_train_test_data(self):
        """
        Create training and test splits (data and corresponding labels) for RNN.
        """
        if self.cached_dataset:
            # return cached intermediate dataset if exists
            try:
                dataset_dict = self._load_dataset(self.cached_dataset, suffix=RNN_SUFFIX)
                return dataset_dict[DATASET_KEY]
            except FileNotFoundError:
                pass

        train_size = len(self.train_data)
        data = self._preprocess_data(self.train_data + self.test_data)
        labels = self._labels_to_prob_vectors(self.train_labels + self.test_labels)

        # perform data embedding through GloVe model
        train_data, test_data = self._get_glove_embedded_data_splits(data, train_size)

        split_dataset = (train_data, test_data, labels[:train_size], labels[train_size:])
        self._dump_dataset(split_dataset, suffix=RNN_SUFFIX, glove_embedding_size=GLOVE_EMBEDDING_SIZE)
        return split_dataset

    def get_spectrum_train_test_data(self):
        """
        Create training and test splits (data and corresponding labels) for Spectrum Kernel.
        """
        if self.cached_dataset:
            # return cached intermediate dataset if exists
            try:
                dataset_dict = self._load_dataset(self.cached_dataset, suffix=SPECTRUM_SUFFIX)
                return dataset_dict[DATASET_KEY]
            except FileNotFoundError:
                pass

        train_size = len(self.train_data)
        data = self._preprocess_data(self.train_data + self.test_data, encode=True)
        labels = self._labels_to_integers(self.train_labels + self.test_labels)

        split_dataset = data[:train_size], data[train_size:], labels[:train_size], labels[train_size:]
        self._dump_dataset(split_dataset, suffix=SPECTRUM_SUFFIX)  # pickle split dataset
        return split_dataset

    def _get_training_inputs_by_labels(self, considered_labels, table_name, inputs_per_label, test_size, random_state):
        """
        Retrieve training pairs given a list of relevant labels.
        :param considered_labels: the list of relevant labels.
        :param table_name: the table where training inputs are stored.
        :param inputs_per_label: how many inputs per label to be retrieved.
        :param test_size: the size of the test split.
        :param random_state: the random state.
        :return: two lists, one containing training data and one containing corresponding labels.
        """
        data = []
        labels = []
        for label in considered_labels:
            label_table = persistence.get_training_inputs_by_label(label, table_name=table_name, limit=inputs_per_label)
            for row in label_table:
                data.append(row[0])
                labels.append(row[1])
        split_dataset = train_test_split(data, labels, test_size=test_size, random_state=random_state)
        self.time = time.time()
        self._dump_dataset(split_dataset)  # pickle split dataset
        return split_dataset

    def _dump_dataset(self, dataset, suffix='', **kwargs):
        """
        Create a dump of the given dataset in secondary storage, appending the given suffix to the filename (to identify
        the intermediate result).
        :param dataset: the object that represents the dataset.
        :param suffix: the string that identifies the intermediate step.
        :param kwargs: a dict that provides extra descriptive parameters of the given dataset.
        """
        dataset_dict = {
            SPECTRUM_KEY: self.spectrum,
            LABELS_KEY: self.considered_labels,
            INPUTS_PER_LABEL_KEY: self.inputs_per_label,
            TIME_KEY: self.time,
            DATASET_KEY: dataset
        }

        if kwargs:
            # merge dicts (with second dict's values overwriting those from the first, if key conflicts exist).
            dataset_dict = {**dataset_dict, **kwargs}

        filename = FILENAME_SEPARATOR.join([str(self.time), str(self.spectrum)] + self.considered_labels + [suffix])
        filename = os.path.join(DATA_FOLDER, filename + DUMP_EXT)
        with open(filename, 'wb') as data_dump:
            pickle.dump(dataset_dict, data_dump)

    @staticmethod
    def _load_dataset(cached_dataset, suffix=None):
        """
        Load a dataset in memory from a dump in secondary storage identified by the given filename and optional suffix 
        (to identify the intermediate result).
        :param cached_dataset: the filename of the dataset.
        :param suffix: the string that identifies the intermediate step.
        :return: the object that represents the dataset.
        """
        if suffix:
            filename = cached_dataset[:-len(DUMP_EXT)] + suffix + DUMP_EXT
        else:
            filename = cached_dataset
        filename = os.path.join(DATA_FOLDER, filename)
        with open(filename, 'rb') as spilt_dataset:
            return pickle.load(spilt_dataset)

    def _preprocess_data(self, data, encode=False):
        # apply shingling on data, each item becomes a shingles list
        preprocessed_data = [SequenceClassifierInput._get_substring(item, spectrum=self.spectrum) for item in data]

        if encode:
            # transform string sequences into binary sequences
            encoded_data = []
            for shingle_list in preprocessed_data:
                encoded_sequence = []
                for shingle in shingle_list:
                    encoded_sequence.append(SequenceClassifierInput._encode_sequence(shingle))
                encoded_data.append(encoded_sequence)
            preprocessed_data = encoded_data

        # pad shingles lists looking at the maximum length
        return SequenceClassifierInput._pad_shingles_lists(preprocessed_data)

    def _labels_to_integers(self, labels):
        """
        Translate the given list of labels into integers (as many as the number of unique labels in the list).
        :param labels: the list of labels.
        :return: the translated list of labels.
        """
        labels_dict = {}
        for i, label in enumerate(self.considered_labels):
            labels_dict[label] = i
        return [labels_dict[label] for label in labels]

    def _labels_to_prob_vectors(self, labels):
        """
        Translate the given list of labels into vectors s.t. all components but one are 0 (as many variants as 
        the number of unique labels in the list).
        :param labels: the list of labels.
        :return: the translated list of labels.
        """
        labels_dict = {}
        num_labels = len(self.considered_labels)
        for i, label in enumerate(self.considered_labels):
            label_vector = [0] * num_labels
            label_vector[i] = 1
            labels_dict[label] = label_vector
        return [labels_dict[label] for label in labels]

    def _get_glove_embedded_data_splits(self, data, train_size):
        """
        Create embeddings of the given data through GloVe model and return train and test splits.
        :param data: the data.
        :param train_size: the size of the train split.
        :return: train and test splits of the given data.
        """
        print('Training GloVe model...')
        glove_model = tf_glove.GloVeModel(embedding_size=GLOVE_EMBEDDING_SIZE, context_size=10)

        start_time = time.time()

        # train GloVe model
        glove_model.fit_to_corpus(data)
        glove_model.train(num_epochs=100)

        elapsed_time = (time.time() - start_time)
        print('GloVe model training time:', timedelta(seconds=elapsed_time))

        # build sequences embeddings and partition the dataset into train and test splits
        glove_matrix = self._build_glove_matrix(glove_model, data)
        train_data = glove_matrix[:train_size]
        test_data = glove_matrix[train_size:]

        print('Training data shape: ', train_data.shape)
        print('Testing data shape:  ', test_data.shape)
        return train_data, test_data

    @staticmethod
    def _get_substring(string, spectrum=3):
        if spectrum == 0:
            result = ['']
        elif len(string) <= spectrum:
            result = [string]
        else:
            result = [string[i: i + spectrum] for i in range(len(string) - spectrum + 1)]
        return result

    @staticmethod
    def _encode_sequence(sequence):
        binary_string = ''.join([AMINO_ACIDS_DICT[amino_acid] for amino_acid in sequence])
        return int(binary_string, BASE_TWO)

    @staticmethod
    def _pad_shingles_lists(data):
        max_length = len(max(data, key=len))

        # pad inputs with respect to max length
        for shingle_list in data:
            padding_length = max_length - len(shingle_list)
            shingle_list += [PADDING_VALUE] * padding_length
        return data

    @staticmethod
    def _build_glove_matrix(glove_model, data):
        """
        Return a matrix which rows correspond to sequences GloVe embeddings.
        Each sequence embedding is computed as the average of the embedding of its n-grams.
        :param glove_model: a trained GloVe model.
        :param data: a list of input data.
        :return: the GloVe embeddings matrix.
        """
        glove_matrix = []
        for shingle_list in data:
            vectors = []
            for shingle in shingle_list:
                vec = glove_model.embedding_for(shingle)
                vectors.append(vec)
            glove_matrix.append(np.mean(vectors, axis=0))  # mean performs better than sum
        return np.asarray(glove_matrix)
