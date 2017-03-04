from domain.domain import Domain
from sklearn.model_selection import train_test_split
import numpy as np


class MachineLearningInput(object):

    def __init__(self):
        self.domain = Domain()

    def get_data_and_labels_by_labels(self,labels):
        train_test_matrix = []
        for label in labels:
            label_table = self.domain.get_sequance_label_data_by_label(label)
            for row in label_table:
                train_test_matrix.append(row)
        train_test_matrix = np.asarray(train_test_matrix)
        return train_test_matrix[:, 0],train_test_matrix[:, 1]

    def get_train_test_data(self,labels,size = 0.25,random_state = 42):
        data, labels = self.get_data_and_labels_by_labels(labels)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(data,labels,test_size = size,random_state = random_state)


if __name__ == '__main__':
    my_domain = MachineLearningInput()
    my_domain.get_train_test_data(["HYDROLASE"])
