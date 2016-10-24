"""
lr_train trains a logistic model using L2 regularization.
This script uses a user selected dataset.
"""
import math
import numpy as np
from scipy.special import expit

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

    norm = lambdaa * W[1:].dot(W[:]) # L2 normalizing
    return loss + norm

def del_nll(W, X, y, lambdaa):
    """
    gradient of nll with respect to weight vector
    """
    n, d = X.shape
    grad = np.zeros(d)

    for i in range(n):
        summand = (s(W.T.dot(X[i]) - y[i])) * X[i]
        grad = grad + summand

    print("lambdaa: {}, weight: {}, un-regularized grad: {}".format(lambdaa, W, grad))
    
    norm = 2 * lambdaa * W # L2 normalizing
    grad.append(norm)

    return grad
