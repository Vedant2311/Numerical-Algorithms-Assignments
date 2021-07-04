import numpy as np

# Gets the Hermitian matrix G such that Inner(p,q) = [p]*G[q]
def constructHermitianG(n):
	G = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if (i+j)%2==0:
				G[i][j] = 2.0/(i+j+1.0)
	return G
	
# Get the inner product of the polynomials
def innerPolynomials(p,q):
	G = constructHermitianG(p.shape[0])
	p = p.reshape((1,p.shape[0]))
	q = q.reshape((q.shape[0],1))
	return (np.matmul(np.conjugate(p),(np.matmul(G,q))))

# Implementation of Modified GS algorithm to get the matrix for the Orthogonal polynomials
def orthogonalizePolynomials(P):

	# Initialising the Q and R matrices
	rows = P.shape[0]
	cols = P.shape[1]	
	Q = np.zeros((rows,cols))
	R = np.zeros((cols,cols))
	
	# Writing the Modified GS pseudocode
	for i in range(cols):
		R[i][i] = np.sqrt(innerPolynomials(P[:,i],P[:,i]))
		Q[:,i] = P[:,i] * 1.0/R[i][i]
		for j in range(i+1,cols):
			R[i][j] = innerPolynomials(Q[:,i],P[:,j])
			P[:,j] = P[:,j] - R[i][j] * Q[:,i]
	return Q

# A function to check whether the implementation is correct or not
# The obtained co-efficients should be multiples of the Legendre Polynomials
def checkCorrectness(n):
	Q = orthogonalizePolynomials(np.eye(n))
	print(Q)

if __name__=='__main__':
	checkCorrectness(5)
