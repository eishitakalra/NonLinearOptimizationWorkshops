
from GenericLineSearchMethod import GLSM
import numpy as np
from ex21func import ex21


tol = 1e-4
func = ex21
x0 = np.array([0.5, 1])  
X = GLSM(x0, func, tol)
Y = np.linalg.norm(X, axis=1)

print(len(Y))
print(f'This is Y {Y}')
print(f'These are the ratios for the first 28 elements of the sequence {(Y[1:27]-0)/(Y[0:26]-0)}')


# Hessian for f(x*)
H_1 = func(2, [0, 0])

# Eigenvalues for the Hessian of f at x*
eig_1 = np.linalg.eigvals(H_1)
print(f'These are the eigenvalues of the Hessian at (0,0) {eig_1}')
if eig_1[0] > 0 and eig_1[1] > 0:
    print(f'Since the eigenvalues of the Hessian of f at point x* = (0,0) are both positive, the hessian is positive definite at x*')
sorted_eig = np.sort(eig_1)[::-1]

# Finding Z = (lambda_n - lambda_1) / (lambda_n + lambda_1) using the eigenvalues
Z = (eig_1[0]-eig_1[1])/(eig_1[0]+eig_1[1])

# The r from Armijo Line Search
# In this case change linesearch in GenericLineSearchMethod.py to Armijo
r = 0.75211265

# The r from exact line search (since the theorem requires SD and Exact)
# In this case change linesearch in GenericLineSearchMethod.py to Exact
# r =  0.79055367

# Check if r is in the set (Z,1)
if r >= Z and r<= 1:
    print(f'r is an element of ({Z},1)')

# Find f at x_k+1 and x_k for sufficiently large k and f at x* = 0
F_0 = func(0, [0, 0])
F_26 = ex21(0,X[26])
F_27 = ex21(0,X[27])

# Find equation F = f(x_k+1) - f(x*)/f(x_k)-f(x*) and r^2
F = ( F_27 - F_0)/(F_26- F_0)
r_sq = r**2

# Check if Theorem 3.4 Nocedal/Wright is matched
if F <= r_sq:
    print(f'Theorem 3.4 is correct')