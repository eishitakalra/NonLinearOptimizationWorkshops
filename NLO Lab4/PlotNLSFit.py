import numpy as np
import matplotlib.pyplot as plt

import doubleexpdecay as ded


# ded.dat holds the data of the points to be fitted
t = ded.dat[:,0]
y = ded.dat[:,1]

fig, ax = plt.subplots()

# plot the data as (blue) points
ax.plot(t, y,'.')

#Copy paste the solutions into the brackets below (comma separated)
r0, c1, c2, l1, l2 = [ 3.58410492, 2.8133427, 2.8133427, 0.69588401, 0.69588949 ] 


fit = r0+c1*np.exp(-l1*t)+c2*np.exp(-l2*t)

# plot the fit as a solid line
ax.plot(t, fit)

plt.show()
