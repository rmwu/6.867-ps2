"""
Problem 3
Implements and runs Pegasos.
"""
import numpy as np

def pegasos(X, y, lambdaa, max_cycles):
    # initialize starting values
    n, d = X.shape
    # incorporate bias term
    w = np.zeros(d+1)
    X = np.c_[np.ones(n), X]
    
    t = 0
    while t < max_cycles:
        # loop through all vectors
        for i in range(n):
            t += 1 # increment time
            eta = 1./(t*lambdaa) # adjusted learning rate
            if y[i]*(w.T.dot(X[i])) < 1:
                # this should be a d+1 by 1
                w0 = w[0]
                w = (1-eta*lambdaa)*w
                w[0] = w0
                w += eta*y[i]*X[i]
            else:
                w0 = w[0]
                w = (1-eta*lambdaa)*w
                w[0] = w0
    print("Updated weight to {} after {} iterations.".format(w, t))
    return w

def pegasos_predict(w):
    # w is extended by bias term
    return lambda x : 1 if w.T.dot(np.insert(x, 0, 1)) > 0 else -1 if w.T.dot(np.insert(x, 0, 1)) < 0 else 0
