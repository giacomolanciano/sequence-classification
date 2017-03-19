from datetime import timedelta

from sklearn import svm
from sklearn import model_selection
import numpy as np
import time

from machine_learning.kernel_functions import precomputed_occurrence_dict_spectrum_kernel
from machine_learning.sequence_classifier import SequenceClassifierInput


clf = svm.SVC(kernel='precomputed')

# build training and test splits
clf_input = SequenceClassifierInput(inputs_per_label=1000)
clf_input.set_train_test_data(['TRANSCRIPTION', 'LYASE'])

# merge splits (cross validation and folding will be performed)
inputs_data = clf_input.train_data + clf_input.test_data
inputs_labels = np.asarray(clf_input.train_labels + clf_input.test_labels)

start_time = time.time()

# pre-compute kernel matrix
kernel_matrix_train = np.asarray(precomputed_occurrence_dict_spectrum_kernel(inputs_data))

# cross validation
param_grid = {'C': [1, 10]}
grid = model_selection.GridSearchCV(clf, param_grid=param_grid, cv=10)
grid.fit(kernel_matrix_train, inputs_labels)

elapsed_time = (time.time() - start_time)

# print stats
print('Best params:   %s' % grid.best_params_)
print('Best accuracy: {:3.2f}%'.format(100 * grid.best_score_))
print('Time: ', timedelta(seconds=elapsed_time))
