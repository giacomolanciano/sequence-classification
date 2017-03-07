from sklearn.model_selection import train_test_split
import numpy as np
from functions import functions
from utils import persistence


class MachineLearningInput(object):
    def __init__(self):
        pass
        #self.train_data, self.test_data, self.train_labels, self.test_labels = None

    def set_train_test_data(self, labels, size=0.25, random_state=42):
        data, labels = MachineLearningInput._get_data_and_labels_by_labels(labels)
        transformed_data = [functions.get_substring(item,3) for item in data]

        self.train_data, self.test_data, self.train_labels, self.test_labels \
            = train_test_split(transformed_data, labels, test_size=size, random_state=random_state)

        for item,label in zip(self.train_data,self.train_labels):
            print(item)
            print(label)

    @staticmethod
    def _get_data_and_labels_by_labels(labels):
        train_test_matrix = []
        for label in labels:
            label_table = persistence.get_training_inputs_by_label(label,limit= 100)
            for row in label_table:
                train_test_matrix.append(row)
        train_test_matrix = np.asarray(train_test_matrix)

        return train_test_matrix[:, 0], train_test_matrix[:, 1]


if __name__ == '__main__':
    my_ml_input = MachineLearningInput()
    my_ml_input.set_train_test_data(['HYDROLASE'])
