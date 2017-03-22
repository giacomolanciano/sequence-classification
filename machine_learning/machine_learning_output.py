import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class MachineLearningOutput():

    def __init__(self,y_test,y_prediction,class_names):
        self.y_prediction = y_prediction
        self.y_test = y_test
        self.confution_matrix = None
        self.class_names = class_names #['TRANSCRIPTION', 'LYASE'] -> 'TRANSCRIPTION' = 0 'LYASE' = 1

    def get_confusion_matrix(self):
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_prediction)


    @staticmethod #cm = confusion_matrix
    def _plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')



if __name__ == '__main__':
    pass
