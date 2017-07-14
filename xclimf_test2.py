import xclimf
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io.mmio import mmread

data = mmread("data/ml_upl5_train.mtx").tocsr()

def is_objective_decreasing_for(D, lbda, gamma, eps=0.01):
    print "=" * 80
    print D, lbda, gamma

    U = 0.01*np.random.random_sample((data.shape[0],D))
    V = 0.01*np.random.random_sample((data.shape[1],D))

    last_objective = -100000

    for i in xrange(15):
        xclimf.update(data, U, V, lbda, gamma)
        objective = xclimf.objective(data, U, V, lbda)
        print "objective: ", objective
        if objective < last_objective + eps:
            print "objective should be bigger or equal last objective..."
            return False
        last_objective = objective 
    return True

Ds = [10,15,20,25]
lbdas = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

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
