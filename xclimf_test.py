import xclimf
import numpy as np
from scipy.sparse import csr_matrix

ra = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
ca = np.array([0, 2, 4, 1, 3, 5, 0, 2, 4, 1, 3, 5])
va = np.array([0.1, 0.4, 0.7,
               0.3, 1., 0.1,
               0.1, 0.4, 0.7,
               0.3, 1., 0.1])

nr = ra.max() + 1
nc = ca.max() + 1

data = csr_matrix((va, (ra, ca)), shape=(nr, nc))

print(data.toarray())

D = 2
lbda = 0.001
gamma = 0.0001

U = 0.01*np.random.random_sample((data.shape[0],D))
V = 0.01*np.random.random_sample((data.shape[1],D))

print(U)
print(V)
last_objective = xclimf.objective(data, U, V, lbda)
print "objective: ", last_objective

for i in xrange(100):
    xclimf.update(data, U, V, lbda, gamma)
    print(U)
    print(V)
    objective = xclimf.objective(data, U, V, lbda)
    print "objective: ", objective
    if objective < last_objective:
        print "objective should be bigger or equal last objective..."
        break
    last_objective = objective    

