from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from utils import persistence

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


def naive_spectrum_kernel(string1, string2):
    kernel_matrix = [[0] * len(string1) for _ in range(len(string2))]
    for row, shingles_list_1 in enumerate(string1):
        for col in range(row, len(string2)):
            shingles_list_2 = string2[col]
            kernel = 0
            for shingle in shingles_list_1:
                if shingle != 0 and shingle in shingles_list_2:
                    kernel += 1
            kernel_matrix[row][col] = kernel
            kernel_matrix[col][row] = kernel
    return kernel_matrix


def dic_spectrum_kernel(X,Y):
    kernel_matrix = [[0] * len(X) for _ in range(len(Y))]
    for row, shingles_list_1 in enumerate(X):
        shingles_list_1_dic = Counter(shingles_list_1)
        for col in range(row, len(Y)):
            shingles_list_2 = Y[col]
            shingles_list_2_dic = Counter(shingles_list_2)
            kernel = 0
            for shingle,occ in shingles_list_1_dic.items():
                try:
                    kernel += shingles_list_2_dic[shingle]*occ
                except KeyError:
                    continue

            kernel_matrix[row][col] = kernel
            kernel_matrix[col][row] = kernel
    print(kernel_matrix)
    return kernel_matrix



class SequenceClassifierInput(object):
    def __init__(self, input_size=1000, progress=True):
        self.progress = progress
        self.input_size = input_size
        self.train_data, self.test_data, self.train_labels, self.test_labels, self.max_feature_size \
            = None, None, None, None, None

    def set_train_test_data(self, considered_labels, size=0.25, random_state=42):
        """
        Create training and test sets for prediction model.
        :param considered_labels: the list of relevant labels.
        :param size: the percentage of input to be used as test set.
        :param random_state: the random state.
        """
        # create label-to-int translation
        labels_dict = {}
        for i, label in enumerate(considered_labels):
            labels_dict[label] = i

        # get training pairs from database
        data, labels = self._get_training_inputs_by_labels(considered_labels)

        # apply shingling on data, each item becomes a shingles list
        data = [SequenceClassifierInput._get_substring(item) for item in data]

        # transform chars sequences in int sequences
        encoded_data = []
        for shingle_list in data:
            transformed_sequence = []
            for shingle in shingle_list:
                transformed_sequence.append(SequenceClassifierInput._encode_sequence(shingle))
            encoded_data.append(transformed_sequence)

        # pad shingles lists looking at the maximum length
        pad_data = SequenceClassifierInput._pad_shingles_lists(encoded_data)
        self.max_feature_size = len(pad_data[0])

        # translate labels in integers
        labels = [labels_dict[label] for label in labels]

        self.train_data, self.test_data, self.train_labels, self.test_labels \
            = train_test_split(pad_data, labels, test_size=size, random_state=random_state)

    def _get_training_inputs_by_labels(self, labels):
        """
        Retrieve training pairs given a list of relevant labels.
        :param labels: the list of relevant labels.
        :return: two lists, one containing training data and one containing corresponding labels.
        """
        train_test_matrix = []
        for label in labels:
            label_table = persistence.get_training_inputs_by_label(label, limit=self.input_size)
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
            shingle_list += [0] * padding_length
        return data


if __name__ == '__main__':
    s1 = 'LUCAMARCHETTI'
    s2 = 'LEONARDOMARTI'
    x = [SequenceClassifierInput._get_substring(s1), SequenceClassifierInput._get_substring(s2)]

    km = dic_spectrum_kernel(x, x)
    for km_row in km:
        print(km_row)
