from GenericLineSearchMethod import GLSM
import numpy as np
from ex21func import ex21
tol = 1e-4
func = ex21
x0 = np.array([0.5, 1])  
X = GLSM(x0, func, tol)
Y = np.linalg.norm(X, axis=1)
print(f'This is Y {Y}')
print(f'These are the ratios for the first 27 elements of the sequence {(Y[1:28]-0)/(Y[0:27]-0)}')
# hessian for f(x*)
H_1 = func(2, [0, 0])
eig_1 = np.linalg.eigvals(H_1)
print(f'These are the eigenvalues of the Hessian at (0,0) {eig_1}')
if eig_1[0] > 0 and eig_1[1] > 0:
    print(f'Since the eigenvalues of the Hessian of f at point x* = (0,0) are both positive, the hessian is positive definite at x*')
sorted_eig = np.sort(eig_1)[::-1]
print(sorted_eig)
Z = (eig_1[0]-eig_1[1])/(eig_1[0]+eig_1[1])
print(Z)
r = 0.752
if r >= Z and r<= 1:
    print(f'r is an element of ({Z},1)')
print(len(Y))
print(Y[27])
F_0 = func(0, [0, 0])
F = (Y[27]- F_0)/(Y[26]- F_0)
r_sq = r**2
if F <= r_sq:
    print(f'Theorem 3.4 is correct')