import tensorflow as tf
import numpy as np
from machine_learning.machine_learning_input import *
import tensorflow.contrib.slim as slim

training_epochs = 25
learning_rate = 0.01
batch_size = 300
display_step = 1
n_class = 3

my_ml_input = MachineLearningInput()
my_ml_input.set_train_test_data(['TRANSCRIPTION', 'LYASE', 'SIGNALING PROTEIN'])

n_attr = my_ml_input.max_feature_size

X_train = np.asarray(my_ml_input.train_data)
y_train = np.asarray(my_ml_input.train_labels)
y_train_new = []

for i in y_train:
    if(i==0):
        y_train_new.append([1,0,0])
    elif(i==1):
        y_train_new.append([0,1,0])
    else:
        y_train_new.append([0,0,1])

y_train_new = np.asarray(y_train_new)

x = tf.placeholder(tf.float32, [None, n_attr])
y = tf.placeholder(tf.int32, [None, n_class])

W = tf.Variable(tf.zeros([n_attr, n_class]))
b = tf.Variable(tf.zeros([n_class]))
print(x)
print(W)

evidence = tf.matmul(x, W) + b

activation = tf.nn.softmax(evidence)
# cross_entropy = (y*tf.log(activation+1))
# cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))

cross_entropy_with_logistic = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = evidence,labels = y))

# loss =  slim.losses.mean_squared_error(evidence, y)
# loss = slim.losses.sparse_softmax_cross_entropy(evidence,y)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_with_logistic)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

avg_set = []
epoch_set=[]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train)/batch_size)+10
        for i in range(total_batch):
            rand_index = np.random.choice(len(X_train),batch_size)
            print(rand_index)
            batch_xs = X_train[rand_index]
            batch_ys = y_train_new[rand_index]
            a = sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            b = sess.run(cross_entropy_with_logistic, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += b/total_batch
        if epoch % display_step == 0:
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(avg_cost))

    print(" Training phase finished")
    correct_prediction = tf.equal(tf.argmax(activation, 1),tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    X_test = my_ml_input.test_data
    y_test = my_ml_input.test_labels
    y_test_new = []
    for i in y_test:
        if (i == 0):
            y_test_new.append([1, 0, 0])
        elif (i == 1):
            y_test_new.append([0, 1, 0])
        else:
            y_test_new.append([0, 0, 1])
    print("MODEL accuracy:", accuracy.eval({x: X_test,y: y_test_new}))