"""
 Generic Line Search Method: applies a generic line search method to
 optimize a given function

 Can choose between different search directions (coordinate descent,
 steepest descent, conjugate gradients) and line search methods
 (backtracking, bisection, exact and pre-determined)

 will return the path taken by the method (so that it can be plotted later)

 Called as path = GLSM(x0, function, tol);

 where
   x0 = starting point
   function = the function to be minimized
          [assumes that the function can be called as
           function(0, x), function(1, x), function(2, x)]
   tol = the convergence tolerance: iterate until \|\nabla f(x)\| < tol

 possible calling sequence is

   import numpy as np
   from rosenbrock import rosenbrock 
   from GenericLineSearchMethod import GLSM
   path = GLSM(x0, rosenbrock, 1e-4)

"""

# ------------------ parameters for the method --------------------------

# direction can be 'CD', 'SD', 'CG' for coordinate descent, steepest descent
# conjugate gradient                               
direction = "SD"

# line search can be 'Exact', 'Armijo', 'Wolfe', '1overk', 'fullstep'
linesearch = "Armijo"

# set line search method parameters
c1 = 0.1  # for Armijo/Backtracking linesearch
c2 = 0.5  # for Bisection linesearch
tol = 1e-7 # tolerance for exact linesearch
max_iter = 100  # iteration limit
do_CG_restarts = False  # restart CG method every n iterations

out = 1 # printing from method: 0=quiet, 1=path, 2=diagnostic

# ------------------ end parameters for the method --------------------------
import numpy as np
from numpy import linalg as LA   # we need this module for the norm
import sys

from LineSearchExact import LineSearchExact as LSExact
from LineSearchBacktrack import LineSearchBacktrack as LSBacktrack
from LineSearchBisection import LineSearchBisection as LSBisection
from CGDirection import CGDirection

def  GLSM(x0, func, eps):
    if (not isinstance(x0, np.ndarray)):
        print("The argument x0 must be a np.array")
        return

    n = x0.size
    # this is to keep a list of iterates
    list = []

    xk = x0
    fk = func(0, xk)
    gk = func(1, xk)
    list.append(np.array(xk))
    tot_neval = 1

    #print(xk)
    if out==2:
        print("Initial g0 = ")
        print(gk)

    if out==1:
        print("f = % 8.5g, |g| = % 8.5g" %(fk, LA.norm(gk)))

    # set these for Conjugate Gradients
    dkprev = None
    gkprev = None

    iter = 0
    while (LA.norm(gk)>=eps and iter<max_iter):
        iter = iter + 1

        # - - - - - - - - - find search directions - - - - - - - - - - 

        if direction=='CD':
            # coordinate descent
            dk = np.eye(1, n, (iter-1)%n).flatten() # i-th coordinate direction
            if (dk.dot(gk)>0):             # make sure it is pointing downwards
                dk = -dk
        elif direction=='SD':
            # steepest descent
            dk = -gk
        elif direction=='CG':
            # conjugate gradient method
            #CG restarts
            if do_CG_restarts and iter%n==1:
                dkprev = None
                gkprev = None
            dk = CGDirection(gk, gkprev, dkprev)
        elif direction=='Newton':
            # Newton Method
            Hk = func(2, xk)
            dk = - np.linalg.solve(Hk, gk)
        else:
            print("Direction code not recognized")
            sys.exit()
            
        # check for descent direction
        if dk.dot(gk)>=0:
            print("search direction is not a descent direction: STOP")
            print(dk)
            sys.exit()
            dk = -gk

        # - - - - - - - - - -  do a line search - - - - - - - - - - 
        if out>1:
            print("dk = ")
            print(dk)

        # counter for number of function evaluations in line search method
        neval = 0

        if linesearch=="Exact":
            # Exact line search:
            alpha, neval = LSExact(xk, dk, tol, func, ret_neval=True)
        elif linesearch=="Armijo":
            # Backtracking Armijo linesearch:
            alpha, neval = LSBacktrack(xk, dk, c1, func, ret_neval=True)
        elif linesearch=="Wolfe":
            # Bisection Wolfe linesearch:
            alpha, neval = LSBisection(xk, dk, c1, c2, func, ret_neval=True)
        elif linesearch=="1overk":
            # predetermined stepsize
            alpha = 1./(LA.norm(dk)*iter)
        elif linesearch=="fullstep":
            alpha = 1
        else:
            print("Linesearch code not recognized")
            sys.exit()

        tot_neval += neval
        if out>=1:
            print("Line search took "+str(neval)+" function evaluation")

        # take step
        xk = xk + alpha*dk


        # remember last dk and gk for conjugate gradients
        gkprev = gk
        dkprev = dk
 
        fk = func(0, xk)
        gk = func(1, xk)
        tot_neval = tot_neval + 1
        
        list.append(np.array(xk))

        if out>=1:
            print("it=% 3d: f = % 8.5g, |g| = % 8.5g" %(iter, fk, LA.norm(gk)));
        if out>1:
            print("xk=")
            print(xk)
            print("gk=")
            print(gk)


    print("GLSM took total of "+str(tot_neval)+" function evaluations")    
    return np.array(list)
