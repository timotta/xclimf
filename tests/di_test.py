from scipy.sparse import csr_matrix
from xclimf import g, dg, precompute_f, precompute_f_optimized
import numpy as np
from datetime import datetime

def di_looping(data, m, U, V, lbda, gamma):
    Vo = V.copy()   
    
    f = precompute_f(data, U, V, m)
     
    for i in f:
        ymi = data[0, i]
        fmi = f[i]
        g_fmi = g(-fmi)
        
        brackets_i = 0
        
        for k in f:
            ymk = data[0, k]
            fmk = f[k]
            fmk_fmi = fmk - fmi
            fmi_fmk = fmi - fmk
                           
            div1 = 1/(1 - (ymk * g(fmk_fmi)))
            div2 = 1/(1 - (ymi * g(fmi_fmk)))
            
            brackets_i += ymk * dg(fmi_fmk) * (div1 - div2)
                
        dI = ymi * (g_fmi + brackets_i) * U[m] - lbda * V[i]
        Vo[i] += dI
        
    return Vo
    
def di_vectors(data, m, U, V, lbda, gamma):
    Vo = V.copy()
    
    (iks, fmi) = precompute_f_optimized(data, U, V, m)
    fmk = fmi.reshape(len(fmi),1)
    
    g_fmi = g(-1 * fmi)
    fmi_fmk = np.subtract(fmi, fmk) 
    fmk_fmi = np.subtract(fmk, fmi) 
         
    ymk = data[0, iks].toarray().transpose()
    ymi = data[0, iks].toarray()
    
    div1 = 1/(1 - (ymk * g(fmk_fmi)))
    div2 = 1/(1 - (ymi * g(fmi_fmk)))
        
    brackets_i = g_fmi + np.sum(ymk * dg(fmi_fmk) * (div1 - div2), axis=0)
    
    dI = (ymi * brackets_i).transpose() * U[m] - lbda * V[iks]

    Vo[iks] += dI
    
    return Vo
        
def test_di_1():
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
    
    df = di_looping(data, 0, U, V, 0.01, 0.01)
    dv = di_vectors(data, 0, U, V, 0.01, 0.01)
    
    print("df", df)
    print("dv", dv)
    print df - dv
    assert np.all(df == dv)
    
def test_di_2():
    v = np.array([0.2,0.5,0.7])
    r = np.array([0,0,0])
    c = np.array([0,2,3])

    data = csr_matrix((v, (r, c)), shape=(2, 6)) 
    
    V = np.array([
      [0.01, 0.02, 0.03, 0.015],
      [0.02, 0.03, 0.05, 0.007],
      [0.05, 0.07, 0.11, 0.101],
      [0.02, 0.01, 0.01, 0.003]
    ])
    
    U = np.array([[0.02, 0.01, 0.03, 0.04]])
    
    df = di_looping(data, 0, U, V, 0.01, 0.01)
    dv = di_vectors(data, 0, U, V, 0.01, 0.01)
    
    print("df", df)
    print("dv", dv)
    print df - dv
    assert np.all(df == dv)
    
def test_di_performance():
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
        di_vectors(data, 0, U, V, 0.01, 0.01)
    time_vector = (datetime.now()-b).total_seconds()
    
    b = datetime.now()
    for i in xrange(1000):
        di_looping(data, 0, U, V, 0.01, 0.01)
    time_looping = (datetime.now()-b).total_seconds()
    
    print "looping", time_looping
    print "vector", time_vector
    assert time_looping > time_vector
    

    



