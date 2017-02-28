import tensorflow as tf
import numpy as np

training_epochs = 25

learning_rate = 0.01

batch_size = 100

display_step = 1

n_attr = 64

n_class = 10

'''
X_train = np.array([[223113, 311311], [131129, 112911],
                    [132528, 252811], [221128, 112811],
                    [132815, 281511], [132529, 252911],
                    [222513, 251311], [221913, 191311],
                    [221524, 152411], [131524, 152411],
                    [131528, 152811], [222222, 222222],
                    [131313, 131313], [131113, 111311],
                    [222522, 252225], [221931, 193130],
                    [221130, 113030], [132513, 251313],
                    [132513, 251325], [132815, 281523],
                    [131129, 112914]]).astype('int32')
y_train = np.array([[1,0],[0,1],
                    [0,1],[1,0],
                    [0,1],[0,1],
                    [1,0],[1,0],
                    [1,0],[0,1],
                    [0,1],[1,0],
                    [0,1],[0,1],
                    [1,0],[1,0],
                    [1,0],[0,1],
                    [0,1],[0,1],
                    [0,1]]).astype('int32')
print(X_train)
print(y_train)'''
from sklearn import datasets
digits = datasets.load_digits()
X_train = digits.data
y_train = digits.target
y_train_new = []

for i in y_train:
    if(i==0):
        y_train_new.append([1,0,0,0,0,0,0,0,0,0])
    elif(i==1):
        y_train_new.append([0,1,0,0,0,0,0,0,0,0])
    elif(i==2):
        y_train_new.append([0,0,1,0,0,0,0,0,0,0])
    elif(i==3):
        y_train_new.append([0,0,0,1,0,0,0,0,0,0])
    elif(i==4):
        y_train_new.append([0,0,0,0,1,0,0,0,0,0])
    elif(i==5):
        y_train_new.append([0,0,0,0,0,1,0,0,0,0])
    elif(i==6):
        y_train_new.append([0,0,0,0,0,0,1,0,0,0])
    elif(i==7):
        y_train_new.append([0,0,0,0,0,0,0,1,0,0])
    elif(i==8):
        y_train_new.append([0,0,0,0,0,0,0,0,1,0])
    elif(i==9):
        y_train_new.append([0,0,0,0,0,0,0,0,0,1])
print(len(X_train[0]))
#print(np.array(y_train_new).astype('int32'))

x = tf.placeholder("float", [None, n_attr])
print(x)

y = tf.placeholder("float", [None, n_class])
print(y)

W = tf.Variable(tf.zeros([n_attr, n_class]))
print(W)

b = tf.Variable(tf.zeros([n_class]))
evidence = tf.matmul(x, W) + b
evidence

activation = tf.nn.softmax(tf.matmul(x, W) + b)
print(activation)

cross_entropy = (y*tf.log(activation+1))
print(cross_entropy)

cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))
print(cost)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
print(optimizer)

avg_set = []

epoch_set=[]

init = tf.global_variables_initializer()
print(init)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train)/batch_size)+9
        o = 0
        for i in range(total_batch):
            batch_xs = X_train[o : o+batch_size]
            batch_ys = y_train_new[o : o+batch_size]
            a = sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            b = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += b/total_batch
            o = o+1
        if epoch % display_step == 0:
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(avg_cost))

    print(" Training phase finished")
    correct_prediction = tf.equal(tf.argmax(activation, 1),tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    X_test = X_train[1700 : 1797]
    y_test = y_train_new[1700 : 1797]
    print("MODEL accuracy:", accuracy.eval({x: X_test,y: y_test}))