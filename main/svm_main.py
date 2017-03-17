from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time

from machine_learning.sequence_classifier import occurrence_dict_spectrum_kernel
from machine_learning.sequence_classifier import SequenceClassifierInput


def print_time(seconds):
    result = '\n'
    if seconds < 60:
        result += 'Time: {:3.2f} seconds'.format(seconds)
    elif (seconds >= 60) and (seconds < 3600):
        result += 'Time: {:3.2f} minutes'.format(seconds / 60)
    else:
        result += 'Time: {:3.2f} hours'.format(seconds / 3600)
    print(result)


clf = svm.SVC(kernel='precomputed')

# build training and test splits
clf_input = SequenceClassifierInput(inputs_per_label=100)
clf_input.set_train_test_data(['TRANSCRIPTION', 'LYASE'])

# merge splits (cross validation and folding will be performed)
inputs_data = clf_input.train_data + clf_input.test_data
inputs_labels = np.asarray(clf_input.train_labels + clf_input.test_labels)

start_time = time.time()

# pre-compute kernel matrix
kernel_matrix_train = np.asarray(occurrence_dict_spectrum_kernel(inputs_data, inputs_data))

# cross validation
param_grid = {'C': [1, 10]}
gd = model_selection.GridSearchCV(clf, param_grid=param_grid, cv=StratifiedKFold(n_splits=10))
gd.fit(kernel_matrix_train, inputs_labels)
print('Best params:   %s' % gd.best_params_)
print('Best accuracy: {:3.2f}%'.format(100 * gd.best_score_))

elapsed = (time.time() - start_time)
print_time(elapsed)
