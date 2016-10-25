import numpy as np
import svm

import argparse

def get_svm_classification_error(Xtr, Ytr, Xval, Yval, C, kernel_func):
    _, predictor = svm.train_svm(Xtr, Ytr, C=C, kernel_func=kernel_func)
    return (np.sum(svm.errors(Xval, Yval, predictor)), predictor)


def find_optimal_parameters(dataset_name,
                            get_error,
                            C_range=[0.01, 0.1, 1, 10, 100],
                            kernel_func=None):
    """
    Find the optimal value of C 
    """
    parts = ("train", "validate", "test")
    datasets = [np.loadtxt('data/data{}_{}.csv'.format(dataset_name, part)) \
        for part in parts]
    (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest) = \
        [(dataset[:, 0:2].copy(), dataset[:, 2:3].copy()) \
            for dataset in datasets]

    def loss_function(C):
        return get_error(Xtrain, Ytrain, Xval, Yval, C, kernel_func)[0]

    def predictor(C):
        return get_error(Xtrain, Ytrain, Xval, Yval, C, kernel_func)[1]

    validation_losses = [loss_function(C_val) for C_val in C_range]
    predictors = [predictor(C_val) for C_val in C_range]
    best_c_index = np.argmin(validation_losses)

    # return best C, validation error on it, and test error
    return (C_range[best_c_index],
            validation_losses[best_c_index],
            np.sum(svm.errors(Xtest, Ytest, predictors[best_c_index])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="name of the dataset")
    args = parser.parse_args()

    def optimize(kernel_func):
        return find_optimal_parameters(args.dataset_name,
                                       get_svm_classification_error,
                                       kernel_func=kernel_func)

    best_lin_c, lin_val_error, lin_test_error = optimize(svm.linear_kernel)

    bandwidths = [0.25, 0.5, 1.0, 2.0]
    rbf_optimization_results = [optimize(svm.make_gaussian_rbf(sigma)) for sigma in bandwidths]
    rbf_val_errors = [x[1] for x in rbf_optimization_results]
    best_sigma_ind = np.argmin(rbf_val_errors)

    best_rbf_c, rbf_val_error, rbf_test_error = rbf_optimization_results[best_sigma_ind]
    best_sigma = bandwidths[best_sigma_ind]

    print("""
        Best linear SVM: C = {}, val error = {}, test error = {}
        Best RBF SVM: C = {}, sigma = {}, val error = {}, test error = {}
        """.format(best_lin_c, lin_val_error, lin_test_error, best_rbf_c, best_sigma, rbf_val_error, rbf_test_error))

if __name__ == '__main__':
    main()
