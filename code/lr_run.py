"""
lr_run.py reads in data and lets the user run
logistic regression, training and testing, as
specified in problem 1.
"""
import csv, sys
import numpy as np

def get_data(filename):
	"""
	get_data(filename) takes in the filename of the
	CSV file and returns an n by m ndarray, containing
	whatever was in the data.
	"""
	with open(filename, "r") as csv_data:
		raw_data = csv.reader(csv_data, delimiter = " ")
		data = []

		for datum in raw_data:
			data.append(datum)

		return np.array(data)

def main():
	"""
	main function
	"""
	if len(sys.argv) != 2:
		print("Usage: %s data_file" % sys.argv[0])
		sys.exit(1)

	data = get_data(sys.argv[1])

if __name__ == "__main__":
	main()