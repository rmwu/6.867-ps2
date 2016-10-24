import math
import numpy as np
import cvxopt

def elementwise_kernel(kernel, vectors, point):
    return np.array([kernel(vector, point) for vector in vectors])

def train_svm(X, Y, C=float('inf'),
              kernel_func=None,
              kernel_mat=None,
              verbose=False):
    """
    Find the optimal parameters alpha_i for the dual form of the C-SVM
    optimization problem.

    X: an n x d array, where each row represents a data point
    Y: an n x 1 vector of labels +-1
    C: the C-SVM regularization parameter
    kernel_func: the kernel function (should take two d x 1 vectors)
    kernel_mat: the kernel matrix (should be n x n)

    Do not specify both a kernel function and kernel matrix.

    The C-SVM dual optimization problem:

    minimize ( 0.5 * sum_{i,j} y^i y^j k(x^i, x^j) alpha_i alpha_j ) + sum_{i} alpha_i
    subject to 0 <= alpha_i <= C (for all i)
    and sum_{i} y^i alpha_i = 0

    The cvxopt matrices are thus

    P_ij = k(x^i, x^j) y^i y^j
    P = (Y.dot(Y.T)) * kernel_matrix
    q = ones

    G = (-1 * identity) stacked on top of identity
    h = 0s, then C

    A = Y.T
    b = 0
    """

    # check preconditions
    if kernel_func and kernel_mat:
        raise ValueError("Specified both kernel function and kernel matrix")

    n = Y.shape[0]  # number of data points


    kernel_matrix = None
    kernel_function = lambda x1, x2: x1.dot(x2)
    if kernel_mat:
        kernel_matrix = kernel_mat
    elif kernel_func:
        kernel_matrix = np.array(
            [kernel_func(X, X[j]) for j in range(n)])
        kernel_function = kernel_func

    else:
        kernel_matrix = X.dot(X.T)  # dot product kernel

    P = cvxopt.matrix(Y.dot(Y.T) * kernel_matrix, tc='d')
    q = cvxopt.matrix(-1 * np.ones(Y.shape))  # n x 1

    G = h = None
    if C == float('inf'):
        G = cvxopt.matrix(-1 * np.identity(n))
        h = cvxopt.matrix(np.zeros(Y.shape))
    else:
        # include additional constraints alpha_i < C
        G = cvxopt.matrix(np.concatenate((-1 * np.identity(n), np.identity(n)), axis=0))
        h = cvxopt.matrix(np.concatenate((np.zeros(Y.shape), C * np.ones(Y.shape)), axis=0), tc='d')

    A = cvxopt.matrix(Y.T, tc='d')
    b = cvxopt.matrix([[0]], tc='d')

    alphas = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])

    # index of a support vector on the margin, exists in practice
    # see http://www.mit.edu/~9.520/spring10/Classes/class05-svm.pdf

    # this line originally read i = np.argmax((alphas < C) * (alphas > 0))
    # however, our alphas vector is not entirely accurate because
    # we never get 0 or C: we get 0 + epsilon or C - epsilon.
    # instead, we pick the closest to C/2, which should solve our woes
    i = np.argmin(np.absolute(alphas  - C/2))

    eps = 1e-6 * C
    clipped_alphas = (alphas > eps) * (alphas < C - eps) * alphas

    optimal_bias = Y[i] - (alphas * Y).T.dot(kernel_function(X, X[i]))

    if verbose:
        optimal_weight = X.T.dot(alphas * Y)
        print("index i of support vector for computation of b, alpha_i: {}" \
            .format(i, alphas[i]))
        print("optimal weight, bias: {}, {}" \
            .format(optimal_weight, optimal_bias))

    # predictor function
    def predictor(new_x):
        return (optimal_bias + (alphas * Y).T.dot(kernel_function(X, new_x)))

    return (alphas, predictor)

def linear_kernel(z1, z2):
    return 1 + z1.dot(z2)

def make_gaussian_rbf(bandwidth):
    """
    Make a Gaussian RBF kernel with given bandwidth.
    """
    def rbf_kernel(z1, z2):
        delta = z1 - z2
        return np.exp(- np.sum(np.atleast_2d(delta * delta), axis=1) / (2 * bandwidth ** 2))

    return rbf_kernel

def main():
    toy_X = np.array([[2, 2], [2, 3], [0, -1], [-3, -2]])
    toy_Y = np.array([[1, 1, -1, -1]]).T

    alphas, predictor = train_svm(toy_X, toy_Y, float('inf'))
    print("alphas: {}".format(alphas))


if __name__ == '__main__':
    main()


