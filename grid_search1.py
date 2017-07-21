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


Ds = [2,4,6,8]
lbdas = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

def is_objective_decreasing_for(D, lbda, gamma):
    print "=" * 80
    print D, lbda, gamma

    U = 0.01*np.random.random_sample((data.shape[0],D))
    V = 0.01*np.random.random_sample((data.shape[1],D))

    print(U)
    print(V)
    last_objective = xclimf.objective(data, U, V, lbda)
    print "objective: ", last_objective

    for i in xrange(15):
        xclimf.update(data, U, V, lbda, gamma)
        print(U)
        print(V)
        objective = xclimf.objective(data, U, V, lbda)
        print "objective: ", objective
        if objective < last_objective:
            print "objective should be bigger or equal last objective..."
            return False
        last_objective = objective 
    return True

ok = []
er = []
for D in Ds:
    for lbda in lbdas:
        for gamma in gammas:
            try:
                if is_objective_decreasing_for(D, lbda, gamma):
                    ok.append((D, lbda, gamma))
            except Exception, e:
                er.append((D, lbda, gamma, e))
                print(e)

print "=" * 80
print "er", er
print "=" * 80
print "ok", ok
print "=" * 80
        
        
    
   

