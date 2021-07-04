import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
from scipy.linalg import hessenberg

# A function that gets the Termination condition for the Eigen value iterations
def term_eigs(H,Q,R,n):
	flag = True
	for i in range(n):
		temp = np.linalg.norm(np.matmul(H,Q[:,i]) - (R[i,i] * Q[:,i]))
		# If norm of any corresponding index is greater than the threshold
		if temp >= 10**-6:
			flag = False
			break
	return flag

## A function that gets the n smallest Eigen values of A
def eigs(A,n):
	# Performing matrix inversion
	A_orig = A.copy()
	A = np.linalg.inv(A)
	
	# Getting the Hessenberg matrix (Tridiagonal in case of Symmetric matrix)
	# Also gets out the reflector Q for the transformation
	H, Q = hessenberg(A, calc_q=True)
	
	# Defining variables for the iteration
	m = A.shape[0]
	Q_eig, R_eig = np.eye(m,m), np.eye(m,m)
	k = 0
	
	while (k<50) and (not (term_eigs(H, Q_eig, R_eig,n))):
		Q_eig, R_eig = np.linalg.qr(H, mode = 'reduced')
		H = np.matmul(R_eig,Q_eig)
		k +=1		
			
	# Getting the eigen values
	lamda = np.zeros((n,1))
	for i in range(n):
		lamda[i] = 1.0/R_eig[i,i]

	# Reducing the eigen vectors to the original basis
	Qhat = np.matmul(Q.T, np.matmul(Q_eig,Q))
	
	return lamda,(Qhat[:,:(n)]/np.linalg.norm(Qhat[:,:(n)]))
	
# A function that checks the working of Eigen QR iterations
def check_eigen_correctness():
	A = np.array([[1,2,4,2,5],[3,5,6,2,5],[6,8,5,4,3],[7,6,5,3,2],[7,5,4,3,2]])
	n = 4
	print(eigs(A,n))

# A function that would compute the 2 norm of a complex number
def complex_norm(temp):
	return math.sqrt(((temp.real)**2) + ((temp.imag)**2))

# A function that computes the value of the function p
def compute_p(a,x0):
	n = a.shape[0]
	temp = complex(0,0)
	z = complex(x0[0],x0[1])
	
	for i in range(n):
		temp += a[i] * (z ** i)
	
	return temp

# A function that computes the value of derivative the function p
def compute_derivative_p(a,x0):
	n = a.shape[0]
	temp = complex(0,0)
	z = complex(x0[0],x0[1])
	
	for i in range(1, n):
		temp += i * a[i] * (z ** (i-1))
	
	return temp

## A function that computes the Jacobian of the function at a given point
def compute_J_p(a,x0):
	temp = compute_derivative_p(a,x0)
	alpha, beta = temp.real, temp.imag
	
	J = np.zeros((2,2))
	J[0,0] = alpha
	J[0,1] = -1.0 * beta
	J[1,0] = beta
	J[1,1] = alpha
	
	return J

# The function that solves Netwon's method
def newton(a,x0):
	k = 0
	temp = compute_p(a,x0)
	xstar = x0
	while k<50 and complex_norm(temp)>=10**-6:
		f_vec = np.array([temp.real, temp.imag])
		xstar = xstar - np.matmul((np.linalg.inv(compute_J_p(a,xstar))),f_vec)
		temp = compute_p(a,xstar)
		k += 1
	return xstar
		
# The function that would make the colored mesh as required
def create_mesh():
	a = np.array([-1.,-1.,0.,1.])		
	x1_list = np.arange(-1.0, 2.05, 0.05)
	x2_list = np.arange(-1.0, 1.05, 0.05)
	
	# Hard-coding the solutions
	sol_1 = np.array([-0.66235, 0.56227])
	sol_2 = np.array([-0.66235, -0.56227])
	sol_3 = np.array([1.32471, 0.0])

	# Starting the plot	
	plt.xlim([-1, 2])
	plt.ylim([-1, 1])
		
	for x1 in x1_list:
		for x2 in x2_list:
			sol = newton(a,np.array([x1,x2]))
			if np.linalg.norm(sol - sol_1) < 10**-3:
				plt.scatter(x1,x2,s=10,color = 'r')
			elif np.linalg.norm(sol - sol_2) < 10**-3:
				plt.scatter(x1,x2,s=10,color = 'b')
			else:
				plt.scatter(x1,x2,s=10,color = 'g')
	
	# The configurations required for the plot
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('Netwon\'s method convergence for various starting points')
	plt.savefig('q6_scatter.png')
	

# The function that checks the correctness of the Newton's algorithm
def check_newton_correctness():
	a = np.array([3.0, -4.0, 1.0])	
	x0 = np.array([1.0, 10.0])
	xstar = newton(a,x0)
	print('The solution for the given polynomial is: ', xstar)

# Here the main function is called
if __name__ == '__main__':
	create_mesh()
	
