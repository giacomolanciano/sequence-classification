from sklearn.model_selection import train_test_split
import numpy as np

from utils import persistence

ALPHABETH_DICT = {"A": 0b10000000000000000000000000,
                  "B": 0b01000000000000000000000000,
                  "C": 0b00100000000000000000000000,
                  "D": 0b00010000000000000000000000,
                  "E": 0b00001000000000000000000000,
                  "F": 0b00000100000000000000000000,
                  "G": 0b00000010000000000000000000,
                  "H": 0b00000001000000000000000000,
                  "I": 0b00000000100000000000000000,
                  "J": 0b00000000010000000000000000,
                  "K": 0b00000000001000000000000000,
                  "L": 0b00000000000100000000000000,
                  "M": 0b00000000000010000000000000,
                  "N": 0b00000000000001000000000000,
                  "O": 0b00000000000000100000000000,
                  "P": 0b00000000000000010000000000,
                  "Q": 0b00000000000000001000000000,
                  "R": 0b00000000000000000100000000,
                  "S": 0b00000000000000000010000000,
                  "T": 0b00000000000000000001000000,
                  "U": 0b00000000000000000000100000,
                  "V": 0b00000000000000000000010000,
                  "W": 0b00000000000000000000001000,
                  "X": 0b00000000000000000000000100,
                  "Y": 0b00000000000000000000000010,
                  "Z": 0b00000000000000000000000001}


def spectrum_kernel(string1, string2):
    kernel_matrix = []
    for ele1 in string1:
        row = []
        for ele2 in string2:
            kernel = 0
            for i in ele1:
                for j in ele2:
                    if i == j and i != 0:
                        kernel += 1
            row.append(kernel)
        kernel_matrix.append(row)
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
    def _encode_sequence(string):
        result = ''
        for element in string:
            result += str(ALPHABETH_DICT[element])
        return int(result)

    @staticmethod
    def _pad_shingles_lists(data):
        max_length = len(max(data, key=len))

        # pad inputs with respect to max length
        for shingle_list in data:
            padding_length = max_length - len(shingle_list)
            shingle_list += [0] * padding_length
        return data


if __name__ == '__main__':
    clf_input = SequenceClassifierInput()
    clf_input.set_train_test_data(['TRANSCRIPTION', 'LYASE', 'SIGNALING PROTEIN'])
