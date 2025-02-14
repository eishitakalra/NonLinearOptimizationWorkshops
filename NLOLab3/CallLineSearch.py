import numpy as np

from chebyquad import chebyquad as func
from GenericLineSearchMethod import GLSM

tol = 1e-4 # tolerance for the line search method

# x0 = np.array([0.33, 0.66])
#x0 = np.array([0.2, 0.4, 0.6])
#x0 = np.array([0.2, 0.4, 0.6, 0.8])
x0 = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
#x0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# path = GLSM(x0, func, tol)
Hk,Sol = GLSM(x0, func, tol)

# print(path)
print(Hk)
print(Sol)


import numpy as np

def compare_hessians(Hk_inv, Sol):
    """Compare the inverse of Hk with the true Hessian at the solution."""
    
    # Compute eigenvalues and eigenvectors
    eigvals_Hk, eigvecs_Hk = np.linalg.eig(Hk_inv)
    eigvals_Sol, eigvecs_Sol = np.linalg.eig(Sol)
    
    print("Eigenvalues of Hk inverse:", eigvals_Hk)
    print("Eigenvalues of Sol (true Hessian):", eigvals_Sol)
    
    # Compute cosine of the angles between corresponding eigenvectors
    angles = []
    for i in range(len(eigvecs_Hk)):
        cos_angle = np.dot(eigvecs_Hk[:, i], eigvecs_Sol[:, i]) / (np.linalg.norm(eigvecs_Hk[:, i]) * np.linalg.norm(eigvecs_Sol[:, i]))
        angles.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))  # Clip to avoid numerical issues
    
    print("Angles (in radians) between corresponding eigenvectors:", angles)
    print("Angles (in degrees) between corresponding eigenvectors:", np.degrees(angles))
    
    return eigvals_Hk, eigvals_Sol, angles


eigvals_Hk, eigvals_Sol, angles = compare_hessians(Hk, Sol)
