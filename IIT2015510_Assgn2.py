"""
    Code to implement regularization in regression
"""
from random import randint
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math

""" Function to normalize data """

def normalize(data):
    m = np.mean(data, axis = 0)
    st_dev = np.std(data, axis = 0)
    return (data - m)/st_dev

def main():

    learning_rate = 1e-2
    epochs = 1500
    lambd = 0.4

    """ Getting data from file """ 

    data = np.loadtxt("ex1data2.txt",dtype=np.float,delimiter=",")

    feature = data[::,0:2]
    price = data[::,-1:]

    fea = normalize(feature)

    features = fea.tolist()
    prices2 = price.tolist()

    prices = [j for i in prices2 for j in i]

    # print (features, prices, sep = '\n')

    n = len(prices)
    dim = 2

    """ Splitting into training and testing data """

    # r = randint(0,45)
    features_train, features_test, prices_train, prices_test = train_test_split(features, prices, test_size=0.25, random_state=6)

    size_train = []
    nbedrooms_train = []

    size_test = []
    nbedrooms_test = []

    for i in range(len(features_train)):
        size_train.append(features_train[i][0])
        nbedrooms_train.append(features_train[i][1])

    for i in range(len(features_test)):
        size_test.append(features_test[i][0])
        nbedrooms_test.append(features_test[i][1])

    # print(size_test,size_train, nbedrooms_test, nbedrooms_train,n,dim)
    # print((prices_test),(prices_train))

    """ Defining weights and model """

    w0 = tf.Variable([1.0], dtype=tf.float32)
    w1 = tf.Variable([1.0], dtype=tf.float32)
    w2 = tf.Variable([1.0], dtype=tf.float32)
    w3 = tf.Variable([1.0], dtype=tf.float32)
    w4 = tf.Variable([1.0], dtype=tf.float32)
    w5 = tf.Variable([1.0], dtype=tf.float32)

    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    model = w0 + w1*x1 + w2*x2 + w3*x1*x1 + w4*x1*x2 + w5*x2*x2

    """ Excluding w0 from regularization """
    cost = tf.reduce_sum(tf.square(model - Y) + lambd*(w1*w1 + w2*w2 + w3*w3 + w4*w4 + w5*w5)) / (2 * len(size_train))
   
    """ Excluding w0, w1 from regularization """
    # cost = tf.reduce_sum(tf.square(model - Y) + lambd*(w2*w2 + w3*w3 + w4*w4 + w5*w5)) / (2 * len(size_train))
   
    """ Excluding w0, w1, w2 from regularization """
    # cost = tf.reduce_sum(tf.square(model - Y) + lambd*(w3*w3 + w4*w4 + w5*w5)) / (2 * len(size_train))
   
    """ Excluding w0, w1, w2, w3 from regularization """
    # cost = tf.reduce_sum(tf.square(model - Y) + lambd*(w4*w4 + w5*w5)) / (2 * len(size_train))
   
    """ Excluding w0, w1, w2, w3, w4 from regularization """
    # cost = tf.reduce_sum(tf.square(model - Y) + lambd*(w5*w5)) / (2 * len(size_train))

    """ No regularization """
    # cost = tf.reduce_sum(tf.square(model - Y) ) / (2 * len(size_train))


    """Training"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(epochs):
        sess.run(optimizer,feed_dict={x1:size_train,x2:nbedrooms_train,Y:prices_train})

    """ Printing weights and Predicting values for test data"""

    wa, wb, wc, wd, we, wf = sess.run([w0, w1, w2, w3, w4, w5], {x1:size_train, x2:nbedrooms_train, Y:prices_train})
    print("w0: %s w1: %s w2: %s w3: %s w4: %s w5: %s "%(wa, wb, wc, wd, we, wf))

    y_ = w0 + w1*x1 + w2*x2 + w3*x1*x1 + w4*x1*x2 + w5*x2*x2 
    pred_y = sess.run(y_, feed_dict={x1: size_test, x2:nbedrooms_test})
    
    for i in range(len(pred_y)):
        print("Predicted: %s, Actual: %s"%(pred_y[i], prices_test[i]))


    """ Calculating and printing RMSE"""

    mse = tf.reduce_mean(tf.square(pred_y - prices_test))
    print("RMSE: %s"%(math.sqrt(sess.run(mse))))

    """ Computing the values of weights from normal equation"""

    print ( "\nWeights using normal equation:\n" )

    X = []
    Y__ = []

    for i in range(35):
        X.append([size_train[i], nbedrooms_train[i]])
        Y__.append(prices_train[i])
    
    X_ = np.matrix(X)
    Y___ = np.matrix(Y__)

    I = np.identity(2)

    xtx = np.matmul(np.transpose(X_),X_)
    xty = np.matmul(np.transpose(X_),np.transpose(Y___))

    lamI = [lambd*x for x in I]

    W = np.matmul(np.linalg.inv(np.add(xtx,lamI)),xty)

    print ( "Weights are : \n", W)


if __name__ == "__main__":
    main()