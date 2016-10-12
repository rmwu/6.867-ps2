"""
6.867 Fall 2016 - Problem Set 1
Problem 1. Implement Gradient Descent
"""
from __future__ import division
import numpy as np
import random

import pylab as pl

debug = False

##################################
# Main Functions
##################################

def gradient_descent(x_init, objective, gradient, eta = 0.000001, threshold = 0.01, 
                     delta = 0.000001, conv_by_grad = False,
                     stochastic = False):
    """
    gradient_descent

    Parameters

        x_init      initial guess
        objective   objective function
        gradient    gradient function
        eta         step size
        threshold   convergence threshold

        conv_by_grad true if converge by grad norm, false otherwise
        stochastic  true if stochastic updates, false by default
    """
    iterations = 0 # counter
    current_x = x_init # this one is updated
    n = x_init.shape[0]

    fx0 = 0 # last f(x)
    fx1 = float("inf") # current f(x)

    grad_norms = []
    delta_objectives = []

    while True:
        # i = iterations % n
        i = random.randrange(n)
        t = iterations # t, as in handout

        if stochastic:
            current_x, grad = update(gradient, current_x, eta, i, t)
        else:
            current_x, grad = update(gradient, current_x, eta)

        current_norm = np.linalg.norm(grad)
        grad_norms.append(current_norm)

        # estimate gradient norm, gradient
        # if stochastic:
        #     est_grad = central_difference(objective, current_x, delta, i)
        # else:
        #     est_grad = central_difference(objective, current_x, delta)
        # est_slope = np.linalg.norm(est_grad)
        
        # calculate objective function
        
        fx1 = sum([objective(current_x, i) for i in range(n)]) if stochastic else objective(current_x)
        
        delta_objectives.append(tuple([current_x, fx1, fx1 - fx0]))
        # delta_objectives.append(fx1 - fx0)

        if debug:
            print("Gradient norm: {}\nCurrent X: {}\nObjective function: {}\nEstimated next gradient: {}"\
                 .format(current_norm, current_x, fx1, est_grad))
            print("Gradient: {}\n".format(grad))
            print("Past objective function: {}\n".format(fx0))

        # check for convergence
        if conv_by_grad and converge_grad(grad, threshold):
            break

        elif not conv_by_grad and converge_delta_fx(fx0, fx1, threshold):
            break
        
        # update "past" objective function
        fx0 = fx1
        iterations += 1

    print("Converged after {} iterations\n".format(iterations))
    # print("We updated to {}\n".format(current_x))

    return (current_x, fx1, iterations, grad_norms, delta_objectives)

def update(gradient, x, eta, i = None, t = None):
    """
    update(gradient, n, current_x, eta, i) returns the
    new x value after updating.

    Parameters
        gradient    gradient function
        x           vector to be updated
        eta         constant step size
        i           optional, to specify stochastic
    """
    step = eta
    if i is not None:
        assert t is not None
        grad = gradient(x, i)
        # step = (eta + t) ** (-0.6) # adjust learning rate
        # print("eta={}\n\n".format(step))
    else:
        grad = gradient(x)

    x_new = x - step * grad

    return (x_new, grad)

##################################
# Estimations
##################################

def central_difference(f, x, delta, stochI=None):
    """
    central_difference(f, x, delta) calculates an approximation of the gradient
    at a given point, for a given function f. The central difference is defined as:

        (f(x + delta/2) - f(x - delta/2)) / delta
    """
    n = x.shape[0]

    delta_matrix = np.identity(n) * delta/2

    est_gradient = []
    for i in range(n):
        ihat = delta_matrix[i].reshape(n)  # unit vector \hat{x}_i
        new_x_pos = x + ihat
        new_x_neg = x - ihat

        if stochI is not None:
            f_pos = f(new_x_pos, stochI)
            f_neg = f(new_x_neg, stochI)
        else:
            f_pos = f(new_x_pos)
            f_neg = f(new_x_neg)

        est_gradient.append((f_pos - f_neg) / delta)
    
    # return (np.linalg.norm(est_gradient), est_gradient)
    return np.array(est_gradient).reshape(n)

def converge_grad(grad, threshold):
    """
    converge_grad(grad, threshold) returns True if the gradient's norm
    is under threshold, False otherwise.

    Parameters
        grad        must be a vector
        threshold   must be a scalar
    """
    return np.linalg.norm(grad) < threshold

def converge_delta_fx(fx0, fx1, threshold):
    """
    converge_delta_fx(fx0, fx1, threshold) returns True if the change in f(x)
    between two success updates is less than threshold, False otherwise.

    Parameters
        fx0         must be a scalar
        fx1         must be a scalar
        threshold   must be a scalar
    """
    return abs(fx1 - fx0) < threshold
