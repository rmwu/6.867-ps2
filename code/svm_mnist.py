import sys

import numpy as np
import svm

import argparse

def get_svm_classification_error(Xtr, Ytr, Xval, Yval, C, kernel_func):
    print("finding error for C = {}".format(C))
    _, predictor = svm.train_svm(Xtr, Ytr, C=C, kernel_func=kernel_func)
    num_val_errors = np.sum(svm.errors(Xval, Yval, predictor))
    print("validation error with C={}: {}/{}".format(C, num_val_errors, len(Yval)))
    return (num_val_errors, predictor)


def find_optimal_parameters(totally_partitioned,
                            get_error,
                            C_range=[1, 10, 100, 300, 1000],
                            kernel_func=None):
    """
    Find the optimal value of C 
    """
    (Xtrain, Ytrain), (Xval, Yval), (Xtest, Ytest) = totally_partitioned

    errorz = [get_error(Xtrain, Ytrain, Xval, Yval, C_val, kernel_func) for C_val in C_range]
    validation_losses = [e[0] for e in errorz]
    predictors = [e[1] for e in errorz]
    best_c_index = np.argmin(validation_losses)

    # return best C, validation error on it, and test error
    return (C_range[best_c_index],
            validation_losses[best_c_index],
            np.sum(svm.errors(Xtest, Ytest, predictors[best_c_index])))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pos", nargs="+")
    parser.add_argument("--neg", nargs="+")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--rbf", action="store_true")
    args = parser.parse_args()

    positive = args.pos
    negative = args.neg

    def make_labeled_examples(digit_subset, label):
        ptrain = []
        pval = []
        ptest = []
        for d in digit_subset:
            examples = np.loadtxt("data/mnist_digit_{}.csv".format(d)).tolist()
            ptrain += examples[:200]
            pval += examples[200:350]
            ptest += examples[350:500]

        def maybe_normalize(arr):
            if args.normalize:
                return (arr * 2/255) - 1
            else:
                return arr

        return (maybe_normalize(np.array(ptrain)), maybe_normalize(np.array(pval)), maybe_normalize(np.array(ptest)))

    ptrain, pval, ptest = make_labeled_examples(positive, 1)
    ntrain, nval, ntest = make_labeled_examples(negative, -1)



    def badly_named_helper(ptrain, ntrain):
        return np.expand_dims(
            np.concatenate(
                (np.ones(len(ptrain)), (-1*np.ones(len(ntrain)))), axis=0), axis=1)

    partitioned = [(np.concatenate(x,axis=0),badly_named_helper(*x)) \
        for x in ((ptrain, ntrain),(pval,nval),(ptest,ntest))]

    print([[x.shape for x in y] for y in partitioned])

    def optimize(kernel_func):
        return find_optimal_parameters(partitioned,
                                       get_svm_classification_error,
                                       kernel_func=kernel_func)

    if args.rbf:
        bandwidths = [3, 30, 300, 3000]
        rbf_optimization_results = [optimize(svm.make_gaussian_rbf(sigma)) for sigma in bandwidths]
        rbf_val_errors = [x[1] for x in rbf_optimization_results]
        best_sigma_ind = np.argmin(rbf_val_errors)

        best_rbf_c, rbf_val_error, rbf_test_error = rbf_optimization_results[best_sigma_ind]
        best_sigma = bandwidths[best_sigma_ind]

        print("Best RBF SVM: C = {}, sigma = {}, val error = {}/{}, test error = {}/{}" \
            .format(best_rbf_c, best_sigma,
                rbf_val_error, len(partitioned[1][1]),
                rbf_test_error, len(partitioned[2][1])))
    else:
        best_lin_c, lin_val_error, lin_test_error = \
            optimize(svm.linear_kernel)
        print("Best linear SVM: C = {}, val error = {}/{}, test error = {}/{}" \
            .format(best_lin_c,
                lin_val_error, len(partitioned[1][1]),
                lin_test_error, len(partitioned[2][1])))

if __name__ == '__main__':
    main()
    print("Command: {}".format(" ".join(sys.argv)))

