import numpy as np
import svm

import argparse

def get_svm_classification_error(Xtr, Ytr, Xval, Yval, C, kernel_func):
    print("finding error for C = {}".format(C))
    _, predictor = svm.train_svm(Xtr, Ytr, C=C, kernel_func=kernel_func)
    num_val_errors = np.sum(svm.errors(Xval, Yval, predictor))
    print("validation error with C={}: {}/{}" \
        .format(C, num_val_errors, len(Yval)))
    return (num_val_errors, predictor)


def find_optimal_parameters(totally_partitioned,
                            get_error,
                            C_range=[0.01, 0.1, 1, 10, 100],
                            kernel_func=None):
    """
    Find the optimal value of C 
    """
    (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest) = totally_partitioned

    errorz = [get_error(Xtrain, Ytrain, Xval, Yval, C_val, kernel_func) \
        for C_val in C_range]
    validation_losses = [e[0] for e in errorz]
    predictors = [e[1] for e in errorz]
    best_c_index = np.argmin(validation_losses)

    # return best C, validation error on it, and test error
    return (C_range[best_c_index],
            validation_losses[best_c_index],
            np.sum(svm.errors(Xtest, Ytest, predictors[best_c_index])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="name of the dataset")
    args = parser.parse_args()

    parts = ("train", "validate", "test")
    datasets = [np.loadtxt('data/data{}_{}.csv' \
        .format(args.dataset_name, part)) \
        for part in parts]

    partitioned = [(dataset[:, 0:2].copy(), dataset[:, 2:3].copy()) \
           for dataset in datasets]

    print([[x.shape for x in y] for y in partitioned])

    def optimize(kernel_func):
        return find_optimal_parameters(partitioned,
                                       get_svm_classification_error,
                                       kernel_func=kernel_func)

    best_lin_c, lin_val_error, lin_test_error = optimize(svm.linear_kernel)

    bandwidths = [0.03, 0.3, 3, 30, 300]
    rbf_optimization_results = [optimize(svm.make_gaussian_rbf(sigma)) \
        for sigma in bandwidths]
    rbf_val_errors = [x[1] for x in rbf_optimization_results]
    best_sigma_ind = np.argmin(rbf_val_errors)

    best_rbf_c, rbf_val_error, rbf_test_error = \
        rbf_optimization_results[best_sigma_ind]
    best_sigma = bandwidths[best_sigma_ind]

    print("Best linear SVM: C = {}, val error = {}, test error = {}" \
        .format(best_lin_c, lin_val_error, lin_test_error))
    print("Best RBF SVM: C = {}, sigma = {}, val error = {}/{}, test error = {}/{}" \
        .format(best_rbf_c, best_sigma,
            rbf_val_error, len(partitioned[1][1]),
            rbf_test_error, len(partitioned[2][1])))

if __name__ == '__main__':
    main()
