import os
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, robust_scale, StandardScaler

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')


def trim_data(x_train, x_test, y_train, y_test):

    x_train = x_train.sort_values("acc_id")
    x_test = x_test.sort_values('acc_id')
    y_train = y_train.sort_values("acc_id")
    y_test = y_test.sort_values('acc_id')
    if not x_train['acc_id'].equals(y_train['acc_id']) or not x_test['acc_id'].equals( y_test['acc_id']):
        print('acc_id doesn.t match')

    x_train = x_train.drop(["acc_id"], axis=1)
    x_test = x_test.drop(['acc_id'], axis=1)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    y_train = y_train.loc[:, ["survival_time"]]
    y_test = y_test.loc[:,['survival_time']]

    return x_train, x_test, y_train, y_test


def scaling_data(x_train, x_test, y_train, y_test):
     x_train = StandardScaler().fit_transform(x_train)
     x_test = StandardScaler().fit_transform(x_test)
     x_test = StandardScaler().fit_transform(x_test)
     x_test = StandardScaler().fit_transform(x_test)

     return x_train, x_test, y_train, y_test


def deep_learning(x_train, x_test, y_train, y_test):
    tf.reset_default_graph()
    # Setting
    epoch, lr = 20000, 0.05
    feature_number = x_train.shape[1]
    node1, node2, node3 = 12, 6 , 3                         #마지막 노드는 무조건 1로
    initializer = tf.contrib.layers.variance_scaling_initializer()  # He
    # initializer = tf.contrib.layers.xavier_initializer()

    X = tf.placeholder(tf.float32, shape=[None, feature_number])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W1 = tf.get_variable("W1", shape=[feature_number, node1], initializer=initializer)
    b1 = tf.Variable(tf.random_normal([node1]))
    W2 = tf.get_variable("W2", shape=[node1, node2], initializer=initializer)
    b2 = tf.Variable(tf.random_normal([node2]))
    W3 = tf.get_variable("W3", shape=[node2, node3], initializer=initializer)
    b3 = tf.Variable(tf.random_normal([1]))
    W4 = tf.get_variable("W4", shape=[node3, 1], initializer=initializer)
    b4 = tf.Variable(tf.random_normal([1]))

    L1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)
    L3 = tf.nn.sigmoid(tf.matmul(L2, W3) + b3)
    hyp = tf.matmul(L3, W4) + b4

    cost = tf.reduce_mean(tf.square(hyp - Y))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
    #train = tf.train.AdagradOptimizer(learning_rate=lr).minimize(cost)
    # train=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, hyp))))

    # %%session run

    print('Learning rate={rate} / Epoch={e}'.format(rate=lr, e=epoch),'->')
    for step in range(epoch + 1):
        train_v, cost_v, hyp_v, Y_v = sess.run([train, cost, hyp, Y], feed_dict={X: x_train, Y: y_train})
        if step % int(epoch / 100) == 0:
            print(' Epoch:', step,"\n", 'Cost:', cost_v,"\n", '-' * 30)
        if cost_v <330:
            break;
    # %%test
    print('Test.....','->')
    predict, true, rmse = sess.run([hyp, Y, rmse], feed_dict={X: x_test, Y: y_test})
    for i in range(100):
        print('Predict:', predict[i], ' -> True:', true[i], )
    print('-> Rmse:', rmse,'\n', '-' * 50)

#----------------------------------------------------------------------------------------------------------------------

x = pd.read_csv('train_xdata.csv')
y = pd.read_csv('train_label.csv')
x = x.sort_values('acc_id')
y = y.sort_values('acc_id')
x_train = x.iloc[:30000,:]
x_test = x.iloc[30000:,:]
y_train = y.iloc[:30000,:]
y_test = y.iloc[30000:,:]

x_train, x_test, y_train, y_test = trim_data(x_train, x_test, y_train, y_test)
x_train, x_test, y_train, y_test = scaling_data(x_train, x_test, y_train, y_test)
deep_learning(x_train, x_test, y_train, y_test)


