from sklearn.model_selection import train_test_split
import numpy as np

from functions import functions
from utils import persistence


class ClassifierInput(object):
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

        # apply shingling on data
        data = [functions.get_substring(item) for item in data]
        transformed_data = []

        # transform chars sequences in int sequences
        for sequence in data:
            transformed_sequence = []
            for shingle in sequence:
                transformed_sequence.append(functions.from_string_to_int(shingle))
            transformed_data.append(transformed_sequence)

        # pad sequences looking at the maximum length
        pad_data, self.max_feature_size = functions.pad_data(transformed_data)

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


if __name__ == '__main__':
    clf_input = ClassifierInput()
    clf_input.set_train_test_data(['TRANSCRIPTION', 'LYASE', 'SIGNALING PROTEIN'])
