from functions.functions import *
from sklearn import svm
from machine_learning.machine_learning_input import *
import time


my_ml_input = MachineLearningInput(input_size=7)  # to speed up testing
my_ml_input.set_train_test_data(['TRANSCRIPTION', 'LYASE', 'SIGNALING PROTEIN'])

clf = svm.SVC(kernel=p_spectrum_kernel_function)
start_time = time.time()
clf.fit(my_ml_input.train_data, np.asarray(my_ml_input.train_labels))
accuracy = clf.score(my_ml_input.test_data, my_ml_input.test_labels)
elapsed = (time.time() - start_time)
if elapsed < 60:
    print('Time: {:3.2f} seconds' .format(elapsed))
elif (elapsed >= 60) and (elapsed <3600):
    print('Time: {:3.2f} minutes'.format(elapsed/60))
else:
    print('Time: {:3.2f} hours'.format(elapsed / 3600))
print('Accuracy: {:3.2f}%'.format(100 * accuracy))

