import xclimf
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io.mmio import mmread

train = mmread("data/ml_upl5_train.mtx").tocsr()
test = mmread("data/ml_upl5_test.mtx").tocsr()

def run(D, lbda, gamma, eps=0.01):
    print "=" * 80
    print D, lbda, gamma

    U = 0.01*np.random.random_sample((train.shape[0],D))
    V = 0.01*np.random.random_sample((train.shape[1],D))

    for i in xrange(25):
        xclimf.update(train, U, V, lbda, gamma)
        objective = xclimf.objective(train, U, V, lbda)
        print "objective: ", objective
        #if i > 4 and objective < last_objective + eps:
        #    print "objective should be bigger or equal last objective..."
        #    return xclimf.compute_mrr(test, U, V)
        #last_objective = objective 
    
    mrr = xclimf.compute_mrr(test, U, V)
    print mrr
    return mrr

Ds = [10,15,20,25]
lbdas = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

ok = []
er = []
for D in Ds:
    for lbda in lbdas:
        for gamma in gammas:
            try:
                mrr = run(D, lbda, gamma)
                ok.append( (mrr, (D, lbda, gamma))  )
            except Exception, e:
                er.append((D, lbda, gamma, e))
                print(e)

print "=" * 80
print "er", er
print "=" * 80
print "ok", ok
print "=" * 80
print sorted(ok, key=lambda a: -a[0])
print "=" * 80


