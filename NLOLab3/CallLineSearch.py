import numpy as np

from chebyquad import chebyquad as func
from GenericLineSearchMethod import GLSM

tol = 1e-4 # tolerance for the line search method

# x0 = np.array([0.33, 0.66])
#x0 = np.array([0.2, 0.4, 0.6])
#x0 = np.array([0.2, 0.4, 0.6, 0.8])
x0 = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
# x0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# path = GLSM(x0, func, tol)
Hk,Sol = GLSM(x0, func, tol)

# print(path)
# print(Hk)
# print(Sol)

Sol_inv = np.linalg.inv(Sol)
Hk_inv = np.linalg.inv(Hk)
# print(Sol_inv)

import numpy as np 

def compare_hessians(Hk, Sol_inv):
    
    print("")
    print("This is the final approximation of inverse of matrix Hk :", Hk)
    print("")
    print("This is the inverse of Hk at the solution: ", Sol_inv)
    print("")

    # Calculate eigenvalues and eigenvectors
    eigvals_Hk, eigvecs_Hk = np.linalg.eig(Hk)
    eigvals_Sol, eigvecs_Sol = np.linalg.eig(Sol_inv)

    # Sort the eigenvalues and keep track of indices
    sorted_indices_Hk = np.argsort(eigvals_Hk)
    sorted_indices_Sol = np.argsort(eigvals_Sol)

    eigvals_Hk_sorted = eigvals_Hk[sorted_indices_Hk]
    eigvecs_Hk_sorted = eigvecs_Hk[:, sorted_indices_Hk]

    eigvals_Sol_sorted = eigvals_Sol[sorted_indices_Sol]
    eigvecs_Sol_sorted = eigvecs_Sol[:, sorted_indices_Sol]

    relative_diff = (eigvals_Hk_sorted - eigvals_Sol_sorted) / eigvals_Sol_sorted
    print("Relative differences in eigenvalues:", relative_diff)
    print("")

    print("Eigenvalues of Hk inverse approximation:", eigvals_Hk_sorted)
    print("")
    print("Eigenvalues of inverse of Hk at Solution :", eigvals_Sol_sorted)
    print("")
    
    # Calculate cosine of the angles between eigenvectors
    angles = []
    costheta = []
    for i in range(len(eigvecs_Hk_sorted)):
        cos_angle = np.dot(eigvecs_Hk_sorted[:,i], eigvecs_Sol_sorted[:,i]) / (np.linalg.norm(eigvecs_Hk_sorted[:,i]) * np.linalg.norm(eigvecs_Sol_sorted[:,i]))
        clipped = np.clip(cos_angle, -1.0, 1.0)
        costheta.append(clipped)
        angles.append(np.arccos(clipped)) 

    print("cos(θ) between eigenvectors :", costheta )
    print("")
    print("Angles (in radians) between corresponding eigenvectors:", angles)
    print("")
    print("Angles (in degrees) between corresponding eigenvectors:", np.degrees(angles))
    
    return eigvals_Hk, eigvals_Sol, angles


eigvals_Hk, eigvals_Sol, angles = compare_hessians(Hk, Sol_inv)

