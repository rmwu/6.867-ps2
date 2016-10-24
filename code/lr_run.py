"""
lr_run.py reads in data and lets the user run
logistic regression, training and testing, as
specified in problem 1.
"""
"""
Logistic regression functions
"""
import numpy as np
from scipy.special import expit
import math

import gradient_descent as gd

def s(x):
    # s(x) is closer to notes
    return expit(x)

def nll(W, X, y, lambdaa):
    """
    y values drawn from {-1, +1}
    """
    n, m = X.shape
    loss = 0

    for i in range(m):
        # scalar
        wTx = W.T.dot(X[i])
        summand = y[i] * math.log(s(wTx)) + (1-y[i]) * math.log(1-s(wTx))
        loss += summand

    norm = lambdaa * W[1:].dot(W[1:]) # L2 normalizing
    return loss + norm

def del_nll(W, X, y, lambdaa):
    """
    gradient of nll with respect to weight vector
    """
    n, d = X.shape
    grad = np.zeros(d)

    for i in range(n):
        summand = (s(W.T.dot(X[i]) - y[i])) * X[i]
        grad += summand
    
    norm = 2 * lambdaa * W # L2 normalizing
    norm[0] = 0
    grad += norm

    return grad

def lr_run(W0, X, y, lambdaa):
    """
    gradient_descent(x_init, objective, gradient, eta = 0.000001, threshold = 0.01, 
                     delta = 0.000001, conv_by_grad = False,
                     stochastic = False)
    gradient_descent returns: (current_x, fx1, iterations, grad_norms, delta_objectives)
    """
    obj = lambda w : nll(w, X, y, lambdaa)
    gradient = lambda w : del_nll(w, X, y, lambdaa)
    eta = 0.0001
    threshold = 0.000001
    stochastic = False

    results = gd.gradient_descent(W0, obj, gradient, eta, threshold, stochastic)
    W = results[0]
    print("Converged to {} from {}.".format(W, W0))
    
    return W
    
def lr_predict(W):
    return lambda x : (1 if s(W.T.dot(x)) > 0.5 else -1)

def lr_verify(X, y, lr_predict):
    n, _ = X.shape
    wrong = 0
    
    for i in range(n):
        if lr_predict(X[i]) != y[i]:
            wrong += 1
    print("{} samples out of {} classified wrong.".format(wrong, n))