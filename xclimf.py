"""
xCLiMF: Optimizing Expected Reciprocal Rank for Data with Multiple Levels of Relevance
Yue Shia, Alexandros Karatzogloub, Linas Baltrunasb, Martha Larsona, Alan Hanjalica
ACM RecSys 2013
"""

from math import exp, log
import numpy as np
import command
import dataset

def g(x):
    """sigmoid function"""
    return 1/(1+exp(-x))

def dg(x):
    """derivative of sigmoid function"""
    return exp(x)/(1+exp(x))**2

def precompute_f(data,U,V,i):
    """precompute f[j] = <U[i],V[j]>
    params:
      data: scipy csr sparse matrix containing user->(item,count)
      U   : user factors
      V   : item factors
      i   : item of interest
    returns:
      dot products <U[i],V[j]> for all j in data[i]
    """
    items = data[i].indices
    f = dict((j,np.dot(U[i],V[j])) for j in items)
    return f
    
def relevance_probability(r, maxi):
  """compute relevance probability as described xClimf paper
  params:
    r:   rating
    ma:  max rating
  """
  return (pow(2,r)-1)/pow(2,maxi)
    
def objective(data,U,V,lbda):
    """compute objective function F(U,V)
    params:
      data: scipy csr sparse matrix containing user->(item,count)
      U   : user factors
      V   : item factors
      lbda: regularization constant lambda
    returns:
      current value of F(U,V)
    """
    maxi = data.max()
    obj = -0.5*lbda*(np.sum(U*U)+np.sum(V*V))
    for m in xrange(len(U)):
        f = precompute_f(data,U,V,m)
        for i in f:
            fmi = f[i]
            ymi = data[m,i]
            rmi = relevance_probability(ymi, maxi)
            brackets = log(g(fmi))
            for j in f:
                fmj = f[j]
                ymj = data[m,j]
                rmj = relevance_probability(ymj, maxi)
                brackets += log(1 - rmj * g(fmj - fmi))
            obj += rmi * brackets 
    return obj / len(U)

def update(data,Uo,Vo,lbda,gamma):
    """update user/item factors using stochastic gradient ascent
    params:
      data : scipy csr sparse matrix containing user->(item,count)
      Uo   : user factors
      Vo   : item factors
      lbda : regularization constant lambda
      gamma: learning rate
    """
    U = Uo.copy()
    V = Vo.copy()
    
    for m in xrange(len(U)):    
        dU = np.zeros(len(U[m]))
        f = precompute_f(data,U,V,m)
        
        for i in f:
            ymi = data[m,i]
            fmi = f[i]
            g_fmi = g(-fmi)
            
            brackets_u = g_fmi * V[i]
            brackets_i = g_fmi
            
            for k in f:
                if i != k:
                    ymk = data[m,k]
                    fmk = f[k]
                    fmk_fmi = fmk - fmi
                    fmi_fmk = fmi - fmk
                    
                    top = ymk * dg(fmk_fmi)
                    bot = 1 - ymk * g(fmk_fmi)
                    sub = V[i] - V[k]
                    brackets_u += top / bot * sub
                    
                    div1 = 1/(1 - (ymk * g(fmk_fmi)))
                    div2 = 1/(1 - (ymi * g(fmi_fmk)))
                    brackets_i += ymk * dg(fmi_fmk) * (div1 - div2)
                
            dI = ymi * brackets_i * U[m] - lbda * V[i]
            Vo[i] += gamma * dI
            
            dU += ymi * brackets_u
            
        dU = dU - lbda * U[m]
        Uo[m] += gamma * dU
      

def compute_mrr(data,U,V):
    """compute average Mean Reciprocal Rank of data according to factors
    params:
      data      : scipy csr sparse matrix containing user->(item,count)
      U         : user factors
      V         : item factors
      the mean MRR over all users in data
    """
    mrr = []
    for m in xrange(data.shape[0]):
        if(len(data[m].indices) > 0):
            items = set(data[m].indices)
            predictions = np.sum(np.tile(U[m],(len(V),1))*V,axis=1)
            for rank,item in enumerate(np.argsort(predictions)[::-1]):
                if item in items:
                    mrr.append(1.0/(rank+1))
                    break
    return np.mean(mrr)

def gradient_ascent(train, test, params, foreach=None, eps=0.1):
    D = params["dims"]
    lbda = params["lambda"]
    gamma = params["gamma"]
    iters = params.get("iters", 25)

    U = 0.01*np.random.random_sample((train.shape[0],D))
    V = 0.01*np.random.random_sample((train.shape[1],D))

    last_objective = float("-inf")

    for i in xrange(iters):
        update(train, U, V, lbda, gamma)
        obj = objective(train, U, V, lbda)
        if foreach:
          foreach(i, obj, U, V, params)
        if obj > last_objective:
           last_objective = obj
        elif obj < last_objective + eps:
            print "objective should be bigger or equal last objective..."
            break

    return (U, V)

def main():
    parser = command.options()
    parser.add_option('--dim',dest='D',type='int',default=10,help='dimensionality of factors (default: %default)')
    parser.add_option('--lambda',dest='lbda',type='float',default=0.001,help='regularization constant lambda (default: %default)')
    parser.add_option('--gamma',dest='gamma',type='float',default=0.0001,help='gradient ascent learning rate gamma (default: %default)')
    parser.add_option('--iters',dest='iters',type='int',default=25,help='max iterations (default: %default)')

    (opts,args) = parser.parse_args()
    if not opts.dataset:
        parser.print_help()
        raise SystemExit
    
    print("reading %s..." % opts.dataset)
    
    (users, items) = dataset.read_users_and_items(opts.dataset, opts.sep, opts.skipfl)

    print("loaded %d users" % len(users))
    print("loaded %d items" % len(items))
    
    topitems = dataset.top_items(items)
    
    print("do not use these top items %s" % str(topitems))
    
    (train, test) = dataset.split_train_test(
      users, topitems, 0.1, 
      opts.topktrain, opts.topktest,
      opts.seltype, opts.norm
    )
    
    def print_mrr(i, objective, U, V, params):
        print("interaction %d: %f" % (i,objective) )
        trainmrr = compute_mrr(train, U, V)
        testmrr = compute_mrr(test, U, V)
        print "train mrr", trainmrr
        print "test mrr", testmrr
    
    params = {
      "dims": opts.D,
      "lambda": opts.lbda,
      "gamma": opts.gamma,
      "iters": opts.iters
    }
    
    (U, V) = gradient_ascent(train, test, params, foreach=print_mrr)
      
    print("U", U)
    print("V", V)
    

if __name__=='__main__':
    main()
