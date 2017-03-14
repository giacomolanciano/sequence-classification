from sklearn.model_selection import train_test_split
import numpy as np

from functions import functions
from utils import persistence


class MachineLearningInput(object):
    def __init__(self, input_size=1000, progress=True):
        self.progress = progress
        self.input_size = input_size
        self.train_data, self.test_data, self.train_labels, self.test_labels, self.max_feature_size = None, None, None, None,None

    def set_train_test_data(self, considered_labels, size=0.25, random_state=42):
        data, labels = self._get_data_and_labels_by_labels(considered_labels)

        # apply shingling on data
        data = [functions.get_substring(item) for item in data]
        transformed_data = []

        for sequence in data:
            transformed_sequence = []
            for shingl in sequence:
                transformed_sequence.append(functions.from_string_to_int(shingl))
            transformed_data.append(transformed_sequence)

        pad_data,self.max_feature_size = functions.pad_data(transformed_data)

        transformed_label = []
        for label in labels:
            if label == 'TRANSCRIPTION':
                transformed_label.append(0)
            elif label == 'LYASE':
                transformed_label.append(1)
            else:
                transformed_label.append(2)

        self.train_data, self.test_data, self.train_labels, self.test_labels \
            = train_test_split(pad_data, transformed_label, test_size=size, random_state=random_state)

        # if self.progress:
        #     for item, label in zip(self.train_data, self.train_labels):
                # print(item)
                # print(label)

    def _get_data_and_labels_by_labels(self, labels):
        train_test_matrix = []
        for label in labels:
            label_table = persistence.get_training_inputs_by_label(label, limit=self.input_size)
            for row in label_table:
                train_test_matrix.append(row)
        train_test_matrix = np.asarray(train_test_matrix)
        return train_test_matrix[:, 0], train_test_matrix[:, 1]


if __name__ == '__main__':
    my_ml_input = MachineLearningInput()
    my_ml_input.set_train_test_data(['TRANSCRIPTION', 'LYASE', 'SIGNALING PROTEIN'])
