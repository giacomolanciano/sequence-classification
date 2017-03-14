from functions.functions import *
from sklearn import svm
from machine_learning.classifier_input import ClassifierInput
import numpy as np
import time


clf_input = ClassifierInput(input_size=10)  # to speed up testing
clf_input.set_train_test_data(['TRANSCRIPTION', 'LYASE', 'SIGNALING PROTEIN'])

clf = svm.SVC(kernel=p_spectrum_kernel_function)
start_time = time.time()
clf.fit(clf_input.train_data, np.asarray(clf_input.train_labels))
accuracy = clf.score(clf_input.test_data, clf_input.test_labels)
elapsed = (time.time() - start_time)

if elapsed < 60:
    print('Time: {:3.2f} seconds' .format(elapsed))
elif (elapsed >= 60) and (elapsed < 3600):
    print('Time: {:3.2f} minutes'.format(elapsed/60))
else:
    print('Time: {:3.2f} hours'.format(elapsed / 3600))

print('Accuracy: {:3.2f}%'.format(100 * accuracy))
