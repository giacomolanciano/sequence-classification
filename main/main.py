from functions.functions import *
from sklearn import svm
from machine_learning.machine_learning_input import *
import time

# X = ["LUCA", "CASA", "CORA", "LARA", "CREA", "COSA", "LOCA", "LICA", "LENA", "CENA", "CERA", "LLLL", "CCCC", "CACA",
#      "LOLO"]
# Y = [1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1]
# X_train = []
#
# # creo X_train il quale conterrà stringhe
#
# for i in X:
#     n_G = get_substring(i, 3)
#     X_train.append(n_G)
#
# # trasformo le stringhe di X_train in numeri in quanto il motodo fit degli estimator lavora solo con numeri,
# # creo l'array train finale X_new_train
#
# X_new_train = []
# for i in X_train:
#     temp = []
#     for j in i:
#         temp.append(f_s_t_f(j))
#     X_new_train.append(temp)
#
# # siccome i componenti di X_new_train hanno lunghezza diversa, si devono rendere uguali, prima trovo il compomente
# # con taglia maggiore....
#
# max_size = 0
# for i in X_new_train:
#     size = len(i)
#     if size > max_size:
#         max_size = size
#
# # ...rendo tutti gli altri della stessa taglia
#
# for i in X_new_train:
#     if len(i) != max_size:
#         for j in range(len(i), max_size):
#             i.append(0)
#
# print(X_new_train)
# # SVM
#
# clf = svm.SVC(kernel=p_spectrum_kernel_function)
# clf.fit(X_new_train, Y)
# k = ["CIAO"]
# k = get_substring(k[0], 3)
#
# Y_test = []
# for i in k:
#     temp = []
#     temp.append(f_s_t_f(i))
#     Y_test.append(temp[0])
# print(Y_test)
# print(len(Y_test))
# print(max_size)
#
# for j in range(len(Y_test), max_size):
#     Y_test.append(0)
#
# print(Y_test)
# print(clf.predict(Y_test))

my_ml_input = MachineLearningInput(input_size=10)  # to speed up testing
my_ml_input.set_train_test_data(['TRANSCRIPTION', 'LYASE', 'SIGNALING PROTEIN'])

clf = svm.SVC(kernel=p_spectrum_kernel_function)
print('---')
print(my_ml_input.train_data)
print(my_ml_input.train_labels)
start_time = time.time()
clf.fit(my_ml_input.train_data, np.asarray(my_ml_input.train_labels))
print('fit completed')
accuracy = clf.score(my_ml_input.test_data, my_ml_input.test_labels)
print("--- %s seconds ---" % (time.time() - start_time))
print(accuracy)

