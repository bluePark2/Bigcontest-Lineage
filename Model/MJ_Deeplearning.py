import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, robust_scale, StandardScaler
import os

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')
scaler = MinMaxScaler()
tf.reset_default_graph()



#x_train/x_test
x_train = pd.read_csv("myWork3.csv")

x_train = x_train.drop(["acc_id"], axis=1)
x_test = x_train.iloc[20000:, :]
x_train = x_train.iloc[0:20000, :]

#Setting
epoch, lr = 10000, 0.01
feature_number, node1, node2,node3 = len(x_train.columns), 20, 10 ,5
initializer = tf.contrib.layers.variance_scaling_initializer() # He
#initializer = tf.contrib.layers.xavier_initializer()

y_train = pd.read_csv("train_label.csv")
y_train = y_train.sort_values("acc_id")
y_train = y_train.reset_index(drop=True)
y_train1 = y_train.loc[:, ["survival_time"]]
y_train = y_train1.iloc[0:20000, :]
y_test = y_train1.iloc[20000:, :]

# Data scailing
#x_train = StandardScaler().fit_transform(x_train)
#x_test = StandardScaler().fit_transform(x_test)
'''
scaler = MinMaxScaler()
x_rb_train = robust_scale(x_train)
x_rb_test = robust_scale(x_test)
x_train = scaler.fit_transform(x_rb_train)
x_test = scaler.fit_transform(x_rb_test)
#Y_data scailing
'''
'''
y_rb_train=robust_scale(y_train)
y_rb_test=robust_scale(y_test)
y_train=scaler.fit_transform(y_rb_train)
y_test=scaler.fit_transform(y_rb_test)
'''

# %%session
X = tf.placeholder(tf.float32, shape=[None, feature_number])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.get_variable("W1", shape=[feature_number, node1], initializer=initializer)
b1 = tf.Variable(tf.random_normal([node1]))

W2 = tf.get_variable("W2", shape=[node1, node2], initializer=initializer)
b2 = tf.Variable(tf.random_normal([node2]))

W3 = tf.get_variable("W3", shape=[node2, 1], initializer=initializer)
b3 = tf.Variable(tf.random_normal([1]))

W4 = tf.get_variable("W4", shape=[node3, 1], initializer=initializer)
b4 = tf.Variable(tf.random_normal([1]))

L1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)
L3 = tf.nn.sigmoid(tf.matmul(L2, W3) + b3)
hyp = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.square(L1 - Y))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
# train=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, hyp))))

# %%session run
print('Case 1 : Learning rate={rate} / Epoch={e}'.format(rate=lr, e=epoch))
print('->')
for step in range(epoch + 1):
    train_v, cost_v, hyp_v, Y_v = sess.run([train, cost, hyp, Y], feed_dict={X: x_train, Y: y_train})
    if step % int(epoch / 100) == 0:
        print('Epoch:', step)
        print('Cost:', cost_v)
        print('-' * 30)
# %%test
print('Test.....\n->')
predict, true, rmse = sess.run([hyp, Y, rmse], feed_dict={X: x_test, Y: y_test})
for i in range(100):
    print('Predict:', predict[i], ' -> True:', true[i], )
print('-> Rmse:', rmse)
print('-' * 50)
# %%
"""
# Show
print('Plotting...')
plt.figure(1,figsize=(4.5,3))
plt.plot(scaler.inverse_transform(x_test[:,0].reshape(-1,1)),
                                  scaler.inverse_transform(y_test[:,0].reshape(-1,1)),'b.')
plt.xlabel('X test set')
plt.ylabel('Y test set')
plt.show()
print('\n','↓'*50,'\n')
plt.figure(2,figsize=(4.5,3))
plt.plot(x_test[:,0],y_test[:,0],'b.')
plt.xlabel('scaled X test set')
plt.ylabel('scaled Y test set')
plt.show()
print('\n','↓'*50,'\n')
plt.figure(3,figsize=(4.5,4.5))
plt.axis([0,1,0,1])
plt.plot(sess.run(hyp,feed_dict={X:x_test}),y_test[:,0],'b.')
plt.xlabel('Hypothesis')
plt.ylabel('scaled Y test set')
plt.show()
"""