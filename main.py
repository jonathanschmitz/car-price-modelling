#!/usr/bin/env python3
"""Implementation of logistic regression & tests"""

import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy
import scipy.io as sio
#import seaborn as sns


def randomize(x, y, length):
    """Randomize order of values in x & y"""
    random_indices = list(range(length))
    random.shuffle(random_indices)
    x = x[random_indices]
    y = y[random_indices]
    return x, y


def prepare_y(y):
    """Make sure y is a vector. Unravel from matrix if necessary (and matrix only
    has 1 column)
    """
    if len(y.shape) > 1:
        if y.shape[1] == 1:
            y = y.ravel()
        else:
            raise ValueError("Y has to be a vector of response values")
    return y


def calc_grad(x, y, theta, lamb=0):
    #import pdb; pdb.set_trace()
    y = prepare_y(y)
    cor_theta = theta.copy()
    m = x.shape[0]  # number of training examples
    h = np.dot(x, theta)  # predicted y values
    cor_theta[0] = 0  # theta_0 does not count for regularization
    grad = np.dot(x.T, (np.subtract(h, y))).T/m + (lamb/m) * cor_theta
    if len(grad.shape) > 1:
        grad = grad[0]
    return grad


def cost(x, y, theta, lamb=0):
    # import pdb; pdb.set_trace()
    y = prepare_y(y)
    cor_theta = theta.copy()
    m = x.shape[0]  # number of training examples
    h = np.dot(x, theta)  # predicted y values
    cor_theta[0] = 0  # theta_0 does not count for regularization
    j = np.sum((np.subtract(h, y) ** 2))/(2*m) + (lamb/(2*m)) * sum(cor_theta**2)
    return j


def normal_equation(train_x, train_y, lamb):
    # normal equation
    reg = np.zeros((train_x.shape[1], train_x.shape[1]))
    np.fill_diagonal(reg, 1)
    reg[0, 0] = 0
    reg = lamb * reg
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(train_x.T, train_x) + reg),
                          train_x.T), train_y)
    return theta


def gradient_descent(x, y, lamb):
    initial_theta = np.ones(x.shape[1]) * 0.1
    cf = lambda t: cost(x, y, t, lamb)
    cf_grad = lambda t: calc_grad(x, y, t, lamb)
    theta = scipy.optimize.fmin_cg(cf, initial_theta, cf_grad, disp=0)
    return theta


def train_parameters(train_x, train_y, cv_x, cv_y):
    # choose lambda
    lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100, 1000, 10000]
    train_errors, cv_errors = [], []
    # train = normal_equation if train_x.shape[1] < 1000 else gradient_descent
    train = gradient_descent
    for lamb in lambdas:
        theta = train(train_x, train_y, lamb)
        train_error = cost(train_x, train_y, theta)
        train_errors.append(train_error)
        cv_error = cost(cv_x, cv_y, theta)
        cv_errors.append(cv_error)
    plt.figure()
    plt.plot(train_errors)
    plt.plot(cv_errors)
    plt.legend(("train", "cv"))
    plt.xticks(range(len(lambdas)), [str(x) for x in lambdas])


def learning_curve(train_x, train_y, cv_x, cv_y, lamb=0):
    """Plot learning curve -> change in training & cv error"""
    train_errors, cv_errors = [], []
    #train = normal_equation if train_x.shape[1] < 1000 else gradient_descent
    train = gradient_descent
    # for n in range(1 if len(train_x) < 20 else len(train_x)// 20,
    #                len(train_x),
    #                len(train_x)//20 if len(train_x) > 20 else 1):
    for n in range(1, len(train_x) + 1):
        theta = train(train_x[:n], train_y[:n], lamb)
        train_error = cost(train_x[:n], train_y[:n], theta)
        import pdb; pdb.set_trace()
        train_errors.append(train_error)
        cv_error = cost(cv_x, cv_y, theta)
        cv_errors.append(cv_error)
    # plt.figure()
    # plt.plot(train_errors)
    # plt.plot(cv_errors)
    # plt.legend(("train", "cv"))
    #plt.show()
    return train_errors, cv_errors


def scale(x):
    #import pdb; pdb.set_trace()
    mu = x.mean(axis=0)
    x = x - mu
    sigma = x.std(axis=0)
    x = x/sigma
    return x, mu, sigma

def add_polynoms(x):
    #import pdb; pdb.set_trace()
    old_x = x.copy()
    for i in range(2, 6):
        x = np.append(x, old_x**i, axis=1)
    return x


def fit_params(x, y):
    """Fit a linear regression to predict response vector y from feature matrix x.
    """
    x, y = np.array(x), np.array(y)
    # add polymonial features
    # feature scaling
    x, mu, s = scale(x)
    #x = add_polynoms(x)
    # first: randomize order
    len_x, len_y = x.shape[0], y.shape[0]
    x = np.insert(x, 0, 1, axis=1)
    if len_x != len_y:
        raise ValueError("Error: features x and response y have different lengths")
    x, y = randomize(x, y, len_x)
    # second: separate into training, cv, and test set
    div1, div2 = math.floor(len_x * 0.6), math.floor(len_x * 0.8)
    train_x, train_y = x[:div1], y[:div1]
    cv_x, cv_y = x[div1:div2], y[div1:div2]
    test_x, test_y = x[div2:], y[div2:]
    learning_curve(train_x, train_y, cv_x, cv_y)
    theta = train_parameters(train_x, train_y, cv_x, cv_y)


def test_descent():
    # test functions using old stuff
    df = pd.read_csv('/home/john/Dokumente/ml-course/ex1/ex1data2.txt', header=None)
    x = df.loc[:, (0, 1)]
    y = df.loc[:, (2)]
    x, mu, s = scale(x)
    x = x.as_matrix()
    x = np.insert(x, 0, 1, axis=1)
    # test unregularized gradient descent:
    assert np.sum(gradient_descent(x, y, 0) - np.array([ 340412.65957447,  110631.0502789 ,   -6649.47427087])) < 0.00001
    assert np.sum(normal_equation(x, y, 0) - np.array([ 340412.65957447,  110631.0502789 ,   -6649.47427087])) < 0.00001


def test_cost_grad():
    # test functions using old stuff
    d = sio.loadmat('/home/john/Dokumente/ml-course/ex5/ex5data1.mat')
    X = np.insert(d["X"], 0, 1, axis=1)
    Xval = np.insert(d["Xval"], 0, 1, axis=1)
    Xtest = np.insert(d["Xtest"], 0, 1, axis=1)
    #cost(X, d["y"].ravel(), np.array([1, 1]), 1)
    #calc_grad(X, d["y"].ravel(), np.array([1, 1]))
    x = np.vstack([X, Xval, Xtest])[:,1]
    y = np.vstack([d["y"], d["yval"], d["ytest"]]).ravel()

    # test cost function
    # assert (cost(X, d["y"].ravel(), np.array([1, 1]), 1) - 303.993192) < (10 ** -5)
    # assert np.sum(calc_grad(X, d["y"], np.array([1, 1]), 1) - np.array([-15.303016, 598.250744])) < 0.00001
    error_train, error_val = learning_curve(X, d["y"], Xval, d["yval"])
    print(error_train)
    print(error_val)

# test stuff here
test_cost_grad()


