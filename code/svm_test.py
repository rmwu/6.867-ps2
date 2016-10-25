import os
import sys

from numpy import *
from plotBoundary import *
import pylab as pl

import svm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="name of the dataset to run on")
parser.add_argument("C", type=float, help="SVM regularization parameter")
parser.add_argument("--save_path", help="directory to save images at, if specified")
parser.add_argument("-v", action="count", help="verbosity")
parser.add_argument("--no_show", action="store_true")

kernel_group = parser.add_mutually_exclusive_group()
kernel_group.add_argument("--rbf", type=float, help="if specified, use RBF kernel with specified bandwidth")
kernel_group.add_argument("--linear", action="store_true", help="use the linear kernel")

args = parser.parse_args()

# parameters
name = args.data_name
C = args.C
rbf_sigma = args.rbf

save_path_train = save_path_val = save_path_test = None
if args.save_path:
	path = args.save_path
	if not os.path.exists(path):
		os.makedirs(path)
	save_path_train = os.path.join(path, "train.pdf")
	save_path_val = os.path.join(path, "validate.pdf")
	save_path_test = os.path.join(path, "test.pdf")

print('======Training======')
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# Carry out training, primal and/or dual
kernel = None
kernel_msg = ""
if rbf_sigma:
	kernel = svm.make_gaussian_rbf(rbf_sigma)
	kernel_msg = "RBF kernel, $\\sigma = {}$".format(rbf_sigma)
if args.linear:
	kernel = svm.linear_kernel
	kernel_msg = "Linear kernel"


print("training SVM {}, C = {}".format(kernel_msg, C))
alphas, predictSVM = svm.train_svm(X, Y, C=C, kernel_func=kernel)

# plot training results
if (not args.no_show) or args.save_path:
	plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1],
		title='SVM Train', save_path=save_path_train)

# print geometry stats
print("geometric margin: {}" \
	.format(1 / np.linalg.norm(svm.optimal_weight_vector(X, Y, alphas))))
print("support vectors / data points: {}/{}" \
	.format(np.sum(alphas > 0), alphas.size))

print('======Validation======')
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]

errors = svm.errors(X, Y, predictSVM)
print("validation error rate: {} = {}/{}" \
	.format(np.mean(errors),
		    np.sum(errors),
		    errors.size))

# plot validation results
if (not args.no_show) or args.save_path:
	plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1],
		title='SVM Validate, {}, $C = {}$'.format(kernel_msg, C),
		save_path=save_path_val)

	validate = loadtxt('data/data'+name+'_validate.csv')
	Xtest = validate[:, 0:2]
	Ytest = validate[:, 2:3]

	plotDecisionBoundary(Xtest, Ytest, predictSVM, [-1, 0, 1],
		title='SVM Test, {}, $C = {}$'.format(kernel_msg, C),
		save_path=save_path_test)

if not args.no_show:
	pl.show()