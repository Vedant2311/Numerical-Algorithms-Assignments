import numpy as np
import matplotlib.pyplot as plt 

## Performs LU factorization with partial pivoting
def lup(A):
	m = A.shape[0]
	U = A.copy()
	L, P = np.eye(m), np.eye(m)
	
	for k in range(m-1):
		# Selects the index of the max element and swaps the required rows
		i = k + np.argmax(abs(U[k:,k]))
		U[[k,i],k:] = U[[i,k],k:]
		L[[k,i],:k] = L[[i,k],:k]
		P[[k,i],:] = P[[i,k],:]
		
		for j in range(k+1,m):
			L[j,k] = U[j,k]/U[k,k]
			U[j,k:] = U[j,k:] - L[j,k] * U[k,k:]
	return P, L, U

## Solves Ax=b using the partial pivoting LU decomposition as done above
def solveLup(P,L,U,b):
	# From Ax=b => PAx = Pb
	m = P.shape[0]
	b = P @ b
	
	# From PAx=b => LUx = b 
	# Taking Ux=w => Lw = b
	# Solving the above equation to get w
	w = np.zeros(m)
	for i in range(m):
		rhs = b[i]
		for j in range(i):
			rhs = rhs - L[i,j] * w[j]
		w[i] = (rhs * 1.)/(L[i,i] * 1.)
	
	# Solving Ux = w to get x 
	x = np.zeros(m)
	i = m-1
	while i>=0:
		rhs = w[i]
		for j in range(i+1,m):
			rhs = rhs - U[i,j] * x[j]
		x[i] = (rhs * 1.)/(U[i,i] * 1.)
		i = i-1
	
	return x.reshape(-1,1)

## Matrix A for which catastrophic rounding results occur
def instabilityMatrix(m):
	A = np.eye(m)
	A[:,m-1] = 1.
	for i in range(1,m):
		A[i,:i] = -1.
	return A

## Performs rook pivoting of the matrix U
def rook_pivot(U,row,col,k):
	row_pivot = k + np.argmax(abs(U[k:,col]))
	col_pivot = k + np.argmax(abs(U[row_pivot,k:]))
	
	if row_pivot==row and col_pivot==col:
		return row,col
	else:
		return rook_pivot(U,row_pivot,col_pivot,k)

## Performs LU factorization with rook pivoting
def lupq(A):
	m = A.shape[0]
	U = A.copy()
	L, P, Q = np.eye(m), np.eye(m), np.eye(m)
	
	for k in range(m-1):
		# Selects the index of the max element as obtained by rook pivoting
		i,l = rook_pivot(U,k,k,k)
		
		# Swaps the k,i rows and k,l columns of U
		U[[k,i],k:] = U[[i,k],k:]
		U[:,[k,l]] = U[:,[l,k]]
		
		# Swaps the k,i rows of L
		L[[k,i],:k] = L[[i,k],:k]
		
		# Swaps the k,i rows of P and k,l columns of Q
		P[[k,i],:] = P[[i,k],:]
		Q[:,[k,l]] = Q[:,[l,k]]
		
		for j in range(k+1,m):
			L[j,k] = U[j,k]/U[k,k]
			U[j,k:] = U[j,k:] - L[j,k] * U[k,k:]
	return P, Q, L, U
	
## Solves Ax=b using the rook pivoting LU decomposition as done above
def solveLupq(P,Q,L,U,b):
	# Making use of the solveLup function to get an x
	x = solveLup(P,L,U,b)	
		
	# Now, making use of the Q matrix to get x=Qx
	x = Q @ x
	return x.reshape(-1,1)

## Generates the plots for growth-factor and backward-error
def get_plots(strin):
	m = range(1,61)
	rho_1, rho_2 = [],[]
	error_1, error_2 = [],[]
	
	for i in m:
		A = instabilityMatrix(i)
		b = np.random.randn(i,1)
		
		P, L, U = lup(A)
		x = (solveLup(P,L,U,b))
		rho_1.append((abs(U).max())/(abs(A).max()))
		error_1.append((np.linalg.norm((A @ x) - b,2))/(np.linalg.norm(b,2)))
		
		P, Q, L, U = lupq(A)
		x = (solveLupq(P, Q, L, U, b))
		rho_2.append((abs(U).max())/(abs(A).max()))
		error_2.append((np.linalg.norm((A @ x) - b,2))/(np.linalg.norm(b,2)))
		
	if strin=='growth':
		plt.plot(m,rho_1,'r',m,rho_2,'b')
		plt.xlabel('The dimensions of the matrix')
		plt.ylabel('The growth factor corresponding to the Instability matrix')
		plt.legend(['LU with partial pivoting','LU with rook pivoting'])
		plt.yscale('log')
		plt.title('Semi-log plot of growth factor v/s matrix dimensions')
		plt.savefig('q6_plot_growth.png')
	else:
		plt.plot(m,error_1,'r',m,error_2,'b')
		plt.xlabel('The dimensions of the matrix')
		plt.ylabel('The relative backward error for solving Ax=b')
		plt.legend(['LU with partial pivoting','LU with rook pivoting'])
		plt.yscale('log')
		plt.title('Semi-log plot of Relative Backward Error v/s matrix dimensions')
		plt.savefig('q6_plot_error.png')
		
		

## Checks for the correctness of the algorithms implemented
def check_correctness():
	A = instabilityMatrix(5)
	b = np.array([1.0, 2.0, 13.0,4.0,-2.0]).reshape(-1,1)
	
	print('Solved with the decomposition: PA = LU')
	P, L, U = lup(A)
	print(solveLup(P,L,U,b))
	print()

	print('Solved with the decomposition: PAQ = LU')
	P, Q, L, U = lupq(A)
	print(solveLupq(P, Q, L, U, b))
	print()

	print('Solved using the numpy solver')
	print(np.linalg.solve(A,b))
	print()
	
if __name__ == '__main__':
	get_plots('error')
