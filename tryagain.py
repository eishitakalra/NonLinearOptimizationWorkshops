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