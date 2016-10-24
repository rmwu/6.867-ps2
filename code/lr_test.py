"""
lr_test.py reads in data and lets the user run
logistic regression, training and testing, as
specified in Problem 1.
"""
import sys

import numpy as np
from plotBoundary import *
import pylab as pl

import gradient_descent as gd
import lr_train as lrt
# import your LR training code

def lr_run(W0, X, y, lambdaa):
    obj = lambda w : lrt.nll(w, X, y, lambdaa)
    gradient = lambda w : lrt.del_nll(w, X, y, lambdaa)

    gd.gradient_descent(W0, obj, gradient, stochastic = False)

def main():
    """
    main function
    """
    if len(sys.argv) != 3:
        print("Usage: %s data_file lambda" % sys.argv[0])
        sys.exit(1)

    # parameters
    filename = sys.argv[1]
    lambdaa = sys.argv[2]

    print('======Training======')
    # load data from csv files
    train = loadtxt(filename)
    X = train[:,0:2]
    Y = train[:,2:3]

    # Carry out training.
    n, _ = Y.shape
    _, d = X.shape

    # reshape to nice vectors
    Y = Y.reshape(n,)

    # generate random starting weights
    W0 = np.random.normal(0, 1, d).reshape(d,)

    lr_run(W0, X, Y, lambdaa)

    # Define the predictLR(x) function, which uses trained parameters
    ### TODO ###

    # plot training results
    plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

    print('======Validation======')
    # load data from csv files
    validate = loadtxt('data/data'+name+'_validate.csv')
    X = validate[:,0:2]
    Y = validate[:,2:3]

    # plot validation results
    plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
    pl.show()

if __name__ == "__main__":
    main()