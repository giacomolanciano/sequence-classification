from sklearn.model_selection import train_test_split
import numpy as np

from utils import persistence
from utils.constants import PADDING_VALUE

BASE_TWO = 2
AMINO_ACIDS_DICT = {'A': '10000000000000000000000000',
                    'B': '01000000000000000000000000',
                    'C': '00100000000000000000000000',
                    'D': '00010000000000000000000000',
                    'E': '00001000000000000000000000',
                    'F': '00000100000000000000000000',
                    'G': '00000010000000000000000000',
                    'H': '00000001000000000000000000',
                    'I': '00000000100000000000000000',
                    'J': '00000000010000000000000000',
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
                    'Z': '00000000000000000000000001'}


class SequenceClassifierInput(object):
    def __init__(self, table_name='protein', inputs_per_label=1000, spectrum=3, progress=True):
        self.progress = progress
        self.table_name = table_name
        self.inputs_per_label = inputs_per_label
        self.spectrum = spectrum
        self.train_data, self.test_data, self.train_labels, self.test_labels, self.max_feature_size \
            = None, None, None, None, None

    def set_train_test_data(self, considered_labels, test_size=0.25, random_state=42):
        """
        Create training and test sets for prediction model.
        :param considered_labels: the list of relevant labels.
        :param test_size: the percentage of input to be used as test set.
        :param random_state: the random state.
        """
        # create label-to-int translation
        labels_dict = {}
        for i, label in enumerate(considered_labels):
            labels_dict[label] = i

        # get training pairs from database
        data, labels = self._get_training_inputs_by_labels(considered_labels)

        # apply shingling on data, each item becomes a shingles list
        data = [SequenceClassifierInput._get_substring(item, spectrum=self.spectrum) for item in data]

        # transform string sequences into int sequences
        encoded_data = []
        for shingle_list in data:
            encoded_sequence = []
            for shingle in shingle_list:
                encoded_sequence.append(SequenceClassifierInput._encode_sequence(shingle))
            encoded_data.append(encoded_sequence)

        # pad shingles lists looking at the maximum length
        pad_data = SequenceClassifierInput._pad_shingles_lists(encoded_data)
        self.max_feature_size = len(pad_data[0])

        # translate labels in integers
        labels = [labels_dict[label] for label in labels]

        self.train_data, self.test_data, self.train_labels, self.test_labels \
            = train_test_split(pad_data, labels, test_size=test_size, random_state=random_state)

    def _get_training_inputs_by_labels(self, labels):
        """
        Retrieve training pairs given a list of relevant labels.
        :param labels: the list of relevant labels.
        :return: two lists, one containing training data and one containing corresponding labels.
        """
        train_test_matrix = []
        for label in labels:
            label_table = persistence.get_training_inputs_by_label(label, table_name=self.table_name,
                                                                   limit=self.inputs_per_label)
            for row in label_table:
                train_test_matrix.append(row)
        train_test_matrix = np.asarray(train_test_matrix)
        return train_test_matrix[:, 0], train_test_matrix[:, 1]

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
