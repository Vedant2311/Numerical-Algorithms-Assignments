import numpy as np
from matplotlib import pyplot as plt

# A function to compute the first n terms of the given sequence
def computeSequence(n):
	diff_seq = np.zeros(n)
	
	# Saving the first two terms of the difference equation
	diff_seq[0] = 1.0/3.0
	diff_seq[1] = 1.0/12.0
	
	# Getting the remaining terms
	for i in range(2,n):
		diff_seq[i] = 2.25 * diff_seq[i-1] - 0.5 * diff_seq[i-2]	
	return diff_seq

# A function to generate the graph for the same
def get_plot(n):
	
	# Get the sequence corresponding to the difference equation
	diff_seq = computeSequence(n+1)
	
	# Get the sequence corresponding to the recurrence relation solution
	rec_seq = np.array([(pow(4.0,-1.0 * k)/3.0) for k in range(n+1)])
	
	# Get the sequence for the values of k 
	k_seq = np.array([k for k in range(n+1)])
	
	# Generating the plots
	plt.plot(k_seq,rec_seq,'r',k_seq,diff_seq,'b')
	plt.xlabel('The values of K')
	plt.ylabel('The solution of the difference equation')
	plt.legend(['Exact solution','Sequence computed'])
	plt.yscale('log')
	plt.title('Semi-log plot of difference equation values v/s Point indices K')
	plt.savefig('q6_plot.png')
	
# Calling the function as asked in the assignment statement
if __name__ == '__main__':
	get_plot(80)
	
	
