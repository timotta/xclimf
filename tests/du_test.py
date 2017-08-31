from scipy.sparse import csr_matrix
from xclimf import g, dg, precompute_f, precompute_f_optimized
import numpy as np
from datetime import datetime

def du_looping(data, m, U, V, lbda, gamma):
    Uo = U.copy()   
    
    dU = np.zeros(len(U[m]))
    f = precompute_f(data, U, V, m)
     
    for i in f:
        ymi = data[0, i]
        fmi = f[i]
        g_fmi = g(-fmi)
        
        brackets_u = g_fmi * V[i]
        
        for k in f:
            ymk = data[0, k]
            fmk = f[k]
            fmk_fmi = fmk - fmi

            top = ymk * dg(fmk_fmi)
            bot = 1 - ymk * g(fmk_fmi)
            sub = V[i] - V[k]
            
            brackets_u += top / bot * sub
                
        dU += ymi * brackets_u
    
    dU = dU - lbda * U[m]
    Uo[m] += gamma * dU
    
    return  Uo
    
def du_vectors(data, m, U, V, lbda, gamma):
    Uo = U.copy()   

    (iks, fmi) = precompute_f_optimized(data, U, V, m)
    fmk = fmi.reshape(len(fmi),1)
    
    fmi_fmk = np.subtract(fmi, fmk) 
    fmk_fmi = np.subtract(fmk, fmi) 
         
    ymk = data[0, iks].toarray().transpose()
    ymi = data[0, iks].toarray()

    top = ymk * dg(fmk_fmi)
    bot = 1 - ymk * g(fmk_fmi)

    viks = V[iks]
    
    vis = np.tile(viks, (1, iks.shape[0])).reshape(iks.shape[0]*iks.shape[0], viks.shape[1])
    vks = np.tile(viks, (iks.shape[0], 1))
    sub = np.subtract(vis, vks)
    
    top_bot = (top / bot)\
      .transpose()\
      .reshape(len(iks) * len(iks), 1)
    
    g_fmi = g(-1 * fmi)
    
    brackets_ui = g_fmi.reshape(len(g_fmi), 1) * V[iks]
    brackets_uk = top_bot * sub
        
    brackets_uk = np.sum(brackets_uk.reshape(len(iks), len(iks), V.shape[1]), axis=1)
    
    brackets_u = brackets_ui + brackets_uk
                
    dU = ymi.transpose() * brackets_u
    dU = np.sum(dU.transpose(), axis=1) - lbda * U[m] 
    
    Uo[m] += (gamma * dU).transpose()
    
    return  Uo
    
def test_du_1():
    v = np.array([0.2,0.5])
    r = np.array([0,0])
    c = np.array([0,2])

    data = csr_matrix((v, (r, c)), shape=(2, 6)) 
    
    V = np.array([
      [0.01, 0.02, 0.03, 0.015],
      [0.02, 0.03, 0.05, 0.007],
      [0.05, 0.07, 0.11, 0.101]
    ])
    
    U = np.array([[0.02, 0.01, 0.03, 0.04]])
    
    dl = du_looping(data, 0, U, V, 0.01, 0.01)
    dv = du_vectors(data, 0, U, V, 0.01, 0.01)
   
    print("dl", dl)
    print("dv", dv)
    print dl - dv
    assert np.all(dl == dv)
    
def test_du_2():
    v = np.array([0.2,0.5,0.7])
    r = np.array([0,0,0])
    c = np.array([0,2,3])

    data = csr_matrix((v, (r, c)), shape=(2, 6)) 
    
    V = np.array([
      [0.01, 0.02, 0.03, 0.015],
      [0.02, 0.03, 0.05, 0.007],
      [0.05, 0.07, 0.11, 0.101],
      [0.02, 0.01, 0.01, 0.03]
    ])
    
    U = np.array([[0.02, 0.01, 0.03, 0.04]])
    
    dl = du_looping(data, 0, U, V, 0.01, 0.01)
    dv = du_vectors(data, 0, U, V, 0.01, 0.01)
   
    print("dl", dl)
    print("dv", dv)
    print dl - dv
    assert np.all(dl == dv)
    
def test_du_performance():
    v = np.array([0.2,0.5,0.7,0.4,0.2])
    r = np.array([0,0,0,0,0])
    c = np.array([0,2,3,4,5])

    data = csr_matrix((v, (r, c)), shape=(2, 6)) 
    
    V = np.array([
      [0.01, 0.02, 0.03, 0.015, 0.02, 0.03, 0.05, 0.007],
      [0.02, 0.03, 0.05, 0.007, 0.01, 0.04, 0.01, 0.021],
      [0.05, 0.07, 0.11, 0.101, 0.02, 0.01, 0.01, 0.003],
      [0.02, 0.01, 0.01, 0.003, 0.02, 0.01, 0.01, 0.003],
      [0.01, 0.04, 0.01, 0.021, 0.02, 0.01, 0.01, 0.003],
      [0.03, 0.02, 0.02, 0.005, 0.05, 0.07, 0.11, 0.101]
    ])
    
    U = np.array([[0.02, 0.01, 0.03, 0.04, 0.02, 0.01, 0.03, 0.04]])
    
    b = datetime.now()
    for i in xrange(1000):
        du_vectors(data, 0, U, V, 0.01, 0.01)
    time_vector = (datetime.now()-b).total_seconds()
    
    b = datetime.now()
    for i in xrange(1000):
        du_looping(data, 0, U, V, 0.01, 0.01)
    time_looping = (datetime.now()-b).total_seconds()
    
    print "looping", time_looping
    print "vector", time_vector
    assert time_looping > time_vector
