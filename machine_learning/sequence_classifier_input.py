import os
import sys

from datetime import timedelta
import numpy as np
from sklearn.model_selection import train_test_split

import klepto
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))
from neural_networks import tf_glove
from utils import persistence
from utils.constants import \
    PADDING_VALUE, SPECTRUM_KEY, LABELS_KEY, INPUTS_PER_LABEL_KEY, TIME_KEY, RNN_SUFFIX, SPECTRUM_SUFFIX, \
    FILENAME_SEPARATOR, DATA_FOLDER, TRAIN_DATA_KEY, TEST_DATA_KEY, TRAIN_LABELS_KEY, TEST_LABELS_KEY, \
    TRAIN_DATA_POS, TEST_DATA_POS, TRAIN_LABELS_POS, TEST_LABELS_POS, GLOVE_TRAIN_SUFFIX, GLOVE_TEST_SUFFIX, \
    GLOVE_EMBEDDING_SIZE_KEY, MAX_COLS_NUM_KEY


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
        """
        A class representing the input data to be fed to a sequence classifier.
        
        :param considered_labels: the list of relevant labels.
        :param cached_dataset: the name of the dataset dump to be restored.
        :param table_name: the table where training inputs are stored.
        :param inputs_per_label: how many inputs per label to be retrieved.
        :param spectrum: the length of a shingle.
        :param test_size: the size of the test split.
        :param random_state: the random state.
        :param progress: whether progress message has to be shown to the user or not.
        """
        self.progress = progress
        self.considered_labels = considered_labels
        self.cached_dataset = cached_dataset
        self.table_name = table_name
        self.inputs_per_label = inputs_per_label
        self.spectrum = spectrum
        self.test_size = test_size
        self.random_state = random_state
        self.time = None
        self.dump_basename = None

        # initialize training and test splits (both data and labels)
        if considered_labels:
            self.cached_dataset = None
            self.labels_num = len(considered_labels)
        elif cached_dataset:
            dataset_dict = self._load_dataset(cached_dataset)
            self.dump_basename = cached_dataset
            self.spectrum = dataset_dict[SPECTRUM_KEY]
            self.considered_labels = dataset_dict[LABELS_KEY]
            self.labels_num = len(self.considered_labels)
            self.inputs_per_label = dataset_dict[INPUTS_PER_LABEL_KEY]
            self.time = dataset_dict[TIME_KEY]
        else:
            raise MissingInputError('Neither labels to be considered nor cached dataset are provided.')

    def get_rnn_train_test_data(self):
        """
        Create training and test splits (data and corresponding labels) for RNN.
        
        :return: the dataset splits to be fed to RNN.
        """
        if self.cached_dataset:
            # return cached intermediate dataset if exists
            try:
                dataset_dict = self._load_dataset(self.cached_dataset, suffix=RNN_SUFFIX)
                train_filename = os.path.join(DATA_FOLDER,
                                              FILENAME_SEPARATOR.join([self.dump_basename, GLOVE_TRAIN_SUFFIX]))
                test_filename = os.path.join(DATA_FOLDER,
                                             FILENAME_SEPARATOR.join([self.dump_basename, GLOVE_TEST_SUFFIX]))

                # retrieve train and test data splits shapes
                train_size = len(dataset_dict[TRAIN_LABELS_KEY])
                test_size = len(dataset_dict[TEST_LABELS_KEY])
                max_cols_num = dataset_dict[MAX_COLS_NUM_KEY]
                glove_embedding_size = dataset_dict[GLOVE_EMBEDDING_SIZE_KEY]

                # restore the memory mappings for train and test data splits
                glove_matrix_train = np.memmap(
                    train_filename, dtype='float32', mode='r+',
                    shape=(train_size, max_cols_num, glove_embedding_size)
                )
                glove_matrix_test = np.memmap(
                    test_filename, dtype='float32', mode='r+',
                    shape=(test_size, max_cols_num, glove_embedding_size)
                )

                return (glove_matrix_train, glove_matrix_test,
                        dataset_dict[TRAIN_LABELS_KEY], dataset_dict[TEST_LABELS_KEY])
            except FileNotFoundError:
                dataset_dict = self._load_dataset(self.cached_dataset)
                train_data = dataset_dict[TRAIN_DATA_KEY]
                test_data = dataset_dict[TEST_DATA_KEY]
                train_labels = dataset_dict[TRAIN_LABELS_KEY]
                test_labels = dataset_dict[TEST_LABELS_KEY]
        else:
            train_data, test_data, train_labels, test_labels = self._get_training_inputs_by_labels()

        train_size = len(train_data)
        data = self._preprocess_data(train_data + test_data)
        train_labels = np.asarray(self._labels_to_prob_vectors(train_labels))
        test_labels = np.asarray(self._labels_to_prob_vectors(test_labels))

        # perform data embedding through GloVe model
        train_data, test_data, max_cols_num = self._get_glove_embedded_data(data, train_size)

        split_dataset = (train_data, test_data, train_labels, test_labels)
        self._dump_dataset(split_dataset, suffix=RNN_SUFFIX,
                           glove_embedding_size=GLOVE_EMBEDDING_SIZE, max_cols_num=max_cols_num)
        return split_dataset

    def get_spectrum_train_test_data(self):
        """
        Create training and test splits (data and corresponding labels) for Spectrum Kernel.
        
        :return: the dataset splits to be fed to SVM (Spectrum Kernel).
        """
        if self.cached_dataset:
            # return cached intermediate dataset if exists
            try:
                dataset_dict = self._load_dataset(self.cached_dataset, suffix=SPECTRUM_SUFFIX)
                return (dataset_dict[TRAIN_DATA_KEY], dataset_dict[TEST_DATA_KEY],
                        dataset_dict[TRAIN_LABELS_KEY], dataset_dict[TEST_LABELS_KEY])
            except FileNotFoundError:
                dataset_dict = self._load_dataset(self.cached_dataset)
                train_data = dataset_dict[TRAIN_DATA_KEY]
                test_data = dataset_dict[TEST_DATA_KEY]
                train_labels = dataset_dict[TRAIN_LABELS_KEY]
                test_labels = dataset_dict[TEST_LABELS_KEY]
        else:
            train_data, test_data, train_labels, test_labels = self._get_training_inputs_by_labels()

        train_size = len(train_data)
        data = self._preprocess_data(train_data + test_data, encode=True, pad=True)
        train_labels = self._labels_to_integers(train_labels)
        test_labels = self._labels_to_integers(test_labels)

        split_dataset = (data[:train_size], data[train_size:], train_labels, test_labels)
        self._dump_dataset(split_dataset, suffix=SPECTRUM_SUFFIX)
        return split_dataset

    def _get_training_inputs_by_labels(self):
        """
        Retrieve training pairs given a list of relevant labels.
        
        :return: two lists containing training data and corresponding labels respectively.
        """
        data = []
        labels = []
        for label in self.considered_labels:
            label_table = persistence.get_training_inputs_by_label(label, table_name=self.table_name,
                                                                   limit=self.inputs_per_label)

            if len(label_table) < self.inputs_per_label:
                raise ValueError('The label %s has less than %d items associated with it in the database.'
                                 % (label, self.inputs_per_label))

            for row in label_table:
                data.append(row[0])
                labels.append(row[1])
        split_dataset = train_test_split(data, labels, test_size=self.test_size, random_state=self.random_state)
        self.time = int(time.time())
        self._dump_dataset(split_dataset)
        return split_dataset

    def _preprocess_data(self, data, encode=False, pad=False):
        """
        Preprocess the data to be fed to a sequence classifier.
        
        :param data: the data to be preprocessed.
        :param encode: whether the binary encoding has to be performed.
        :param pad: whether the shingles lists padding has to be performed.
        :return: the preprocessed data.
        """
        # apply shingling on data, each item becomes a shingles list
        data = [self._get_n_grams(item, n=self.spectrum) for item in data]

        if encode:
            # transform string sequences into binary sequences
            encoded_data = []
            for shingle_list in data:
                encoded_sequence = []
                for shingle in shingle_list:
                    encoded_sequence.append(self._encode_sequence(shingle))
                encoded_data.append(encoded_sequence)
            data = encoded_data

        if pad:
            # pad shingles lists looking at the maximum length
            data = self._pad_shingles_lists(data)

        return data

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

    def _get_glove_embedded_data(self, data, train_size):
        """
        Compute two matrices which rows correspond to sequences GloVe embeddings of train and test splits respectively.
        Each sequence embedding is computed as the sequence of the embedding of its n-grams.

        :param data: a list of input data.
        :param train_size: the size of the training split.
        :return: the GloVe embeddings matrices for train and test splits respectively and the max sequence length.
        """
        max_cols_num = len(max(data, key=len))
        train_filename = os.path.join(DATA_FOLDER, FILENAME_SEPARATOR.join([self.dump_basename, GLOVE_TRAIN_SUFFIX]))
        test_filename = os.path.join(DATA_FOLDER, FILENAME_SEPARATOR.join([self.dump_basename, GLOVE_TEST_SUFFIX]))

        glove_matrix_train = np.memmap(train_filename, dtype='float32', mode='w+',
                                       shape=(train_size, max_cols_num, GLOVE_EMBEDDING_SIZE))
        glove_matrix_test = np.memmap(test_filename, dtype='float32', mode='w+',
                                      shape=(len(data) - train_size, max_cols_num, GLOVE_EMBEDDING_SIZE))

        glove_model = self._train_glove_model(data)

        # build sequences embeddings and partition the dataset into train and test splits
        for idx, shingle_list in enumerate(data):
            embeddings = [glove_model.embedding_for(shingle) for shingle in shingle_list]

            # pad the sequence with respect to max length
            padding_length = max_cols_num - len(embeddings)
            embeddings += [[PADDING_VALUE] * GLOVE_EMBEDDING_SIZE] * padding_length

            if idx < train_size:
                glove_matrix_train[idx] = np.asarray(embeddings)
                glove_matrix_train.flush()
            else:
                glove_matrix_test[idx - train_size] = np.asarray(embeddings)
                glove_matrix_test.flush()
        return glove_matrix_train, glove_matrix_test, max_cols_num

    def _dump_dataset(self, dataset, suffix='', **kwargs):
        """
        Create a dump of the given dataset in secondary storage, appending the given suffix to the filename (to identify
        the intermediate result). The dataset must be a tuple of four elements corresponding respectively to:
        train data, test data, train labels, test labels.

        :type dataset: tuple
        :param dataset: the object that represents the dataset.
        :param suffix: the string that identifies the intermediate step.
        :param kwargs: a dict that provides extra descriptive parameters of the given dataset.
        """
        dataset_dict = {
            SPECTRUM_KEY: self.spectrum,
            LABELS_KEY: self.considered_labels,
            INPUTS_PER_LABEL_KEY: self.inputs_per_label,
            TIME_KEY: self.time,
            TRAIN_LABELS_KEY: dataset[TRAIN_LABELS_POS],
            TEST_LABELS_KEY: dataset[TEST_LABELS_POS]
        }

        if suffix != RNN_SUFFIX:
            data_dict = {
                TRAIN_DATA_KEY: dataset[TRAIN_DATA_POS],
                TEST_DATA_KEY: dataset[TEST_DATA_POS]
            }
            dataset_dict.update(data_dict)

        if kwargs:
            # merge dicts (with second dict's values overwriting those from the first, if key conflicts exist).
            dataset_dict.update(kwargs)

        if not self.dump_basename:
            self.dump_basename = FILENAME_SEPARATOR.join(
                [str(self.time), str(self.spectrum), str(self.inputs_per_label)] + self.considered_labels
            )

        if suffix != '':
            dirname = FILENAME_SEPARATOR.join([self.dump_basename, suffix])
        else:
            dirname = self.dump_basename

        dirname = os.path.join(DATA_FOLDER, dirname)
        archive = klepto.archives.dir_archive(dirname, cached=True, serialized=True)
        for key, val in dataset_dict.items():
            archive[key] = val
        try:
            archive.dump()
        except MemoryError:
            print('The dataset dump %s has not been stored due to memory error.' % dirname, file=sys.stderr)

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
            dirname = FILENAME_SEPARATOR.join([cached_dataset, suffix])
        else:
            dirname = cached_dataset
        dirname = os.path.join(DATA_FOLDER, dirname)

        if not os.path.isdir(dirname):
            raise FileNotFoundError

        archive = klepto.archives.dir_archive(dirname, cached=True, serialized=True)
        archive.load()
        return archive

    @staticmethod
    def _train_glove_model(data):
        """
        Train a GloVe model with the given data.
        
        :param data: the data.
        :return: the trained GloVe model.
        """
        print('Training GloVe model...')
        glove_model = tf_glove.GloVeModel(embedding_size=GLOVE_EMBEDDING_SIZE, context_size=10, max_vocab_size=1000000)

        start_time = time.time()

        # train GloVe model
        glove_model.fit_to_corpus(data)
        glove_model.train(num_epochs=100)

        elapsed_time = (time.time() - start_time)
        print('GloVe model training time:', timedelta(seconds=elapsed_time))
        return glove_model

    @staticmethod
    def _get_n_grams(string, n=3):
        if n == 0:
            result = ['']
        elif len(string) <= n:
            result = [string]
        else:
            result = [string[i: i + n] for i in range(len(string) - n + 1)]
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


if __name__ == '__main__':
    CONSIDERED_CLASSES = ['HYDROLASE', 'TRANSFERASE']
    clf_input = SequenceClassifierInput(cached_dataset='1494918744_3_HYDROLASE_TRANSFERASE')
    clf_input.get_rnn_train_test_data()
