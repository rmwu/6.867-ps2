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
        # update step
        current_x, grad = update(gradient, current_x, eta)

        current_norm = np.linalg.norm(grad)
        grad_norms.append(current_norm)
        
        # calculate objective function
        fx1 = objective(current_x)
        
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
    
    print("Converged after {} iterations.".format(iterations))
    return (current_x, grad_norms, delta_objectives)

def update(gradient, x, eta):
    """
    update(gradient, n, x, eta) returns the
    new x value after updating.

    Parameters
        gradient    gradient function
        x           vector to be updated
        eta         constant step size
    """
    grad = gradient(x)
    x_new = x - eta * grad

    return (x_new, grad)

##################################
# Estimations
##################################

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
