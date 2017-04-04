from datetime import timedelta

from sklearn import svm
from sklearn import model_selection
import numpy as np
import time

from machine_learning.kernel_functions import precomputed_occurrence_dict_spectrum_kernel
from machine_learning.kernel_functions import occurrence_dict_spectrum_kernel
from machine_learning.model_performance_measure import ModelPerformanceMeasure
from machine_learning.sequence_classifier import SequenceClassifierInput

CONSIDERED_CLASSES = ['OXIDOREDUCTASE', 'PROTEIN TRANSPORT', 'LECTIN']
# CONSIDERED_CLASSES = ['ALPHA', 'BETA']


clf = svm.SVC(kernel='precomputed')

# build training and test splits
print('Loading dataset...')
clf_input = SequenceClassifierInput(CONSIDERED_CLASSES, table_name='protein', inputs_per_label=200, spectrum=3)
train_data, test_data, train_labels, test_labels = clf_input.get_spectrum_train_test_data()
train_labels = np.asarray(train_labels)

start_time = time.time()

# pre-compute kernel matrix
print('Pre-computing kernel matrix...')
kernel_matrix_train = np.asarray(precomputed_occurrence_dict_spectrum_kernel(train_data))

# cross validation
print('Performing k-fold cross validation...')
param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
grid = model_selection.GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)
grid.fit(kernel_matrix_train, train_labels)

elapsed_time = (time.time() - start_time)

# print stats
print('\nBest params:   %s' % grid.best_params_)
print('Best accuracy: {:3.2f}%'.format(100 * grid.best_score_))
print('Time: ', timedelta(seconds=elapsed_time))

# show confusion matrix for best params
print('\nComputing confusion matrix for best params...')
clf = svm.SVC(kernel=occurrence_dict_spectrum_kernel, C=grid.best_params_['C'])
clf.fit(train_data, train_labels)
pred_labels = clf.predict(test_data)
performance_measure = ModelPerformanceMeasure(test_labels, pred_labels, CONSIDERED_CLASSES)
performance_measure.build_confusion_matrix()
performance_measure.plot_confusion_matrix()
