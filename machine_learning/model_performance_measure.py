import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class ModelPerformanceMeasure:
    def __init__(self, y_test, y_prediction, class_labels):
        self.y_prediction = y_prediction
        self.y_test = y_test
        self.confusion_matrix = None
        self.class_labels = class_labels  # ['TRANSCRIPTION', 'LYASE'] -> 'TRANSCRIPTION' = 0 'LYASE' = 1

    def build_confusion_matrix(self):
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_prediction)

    def plot_confusion_matrix(self, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        plt.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.class_labels))
        plt.xticks(tick_marks, self.class_labels, rotation=45)
        plt.yticks(tick_marks, self.class_labels)

        if normalize:
            self.confusion_matrix = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
            self.confusion_matrix = np.around(self.confusion_matrix, decimals=2)
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(self.confusion_matrix)

        thresh = self.confusion_matrix.max() / 2.
        for i, j in itertools.product(range(self.confusion_matrix.shape[0]), range(self.confusion_matrix.shape[1])):
            plt.text(j, i, self.confusion_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if self.confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


if __name__ == '__main__':
    pass
