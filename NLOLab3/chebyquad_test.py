import numpy as np

from chebyquad import chebyquad as func

x0 = np.array([0.33, 0.66])
f = func(0,x0)
g = func(1,x0)
H = func(2,x0)
print(f)
print(g)
print(H)
