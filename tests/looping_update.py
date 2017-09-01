import numpy as np
from math import exp, log

def g(x):
    """sigmoid function"""
    return 1/(1+exp(-x))

def dg(x):
    """derivative of sigmoid function"""
    return exp(x)/(1+exp(x))**2

def precompute_f(data,U,V,m):
    """precompute f[j] = <U[m],V[j]>
    params:
      data: scipy csr sparse matrix containing user->(item,count)
      U   : user factors
      V   : item factors
      m   : user of interest
    returns:
      dot products <U[m],V[j]> for all j in data[i]
    """
    items = data[m].indices
    f = dict((j,np.dot(U[m],V[j])) for j in items)
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
