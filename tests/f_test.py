from scipy.sparse import csr_matrix
import xclimf
import looping_update
import numpy as np
from datetime import datetime

def test_f_1():
    v = np.array([0.2,0.3,0.5])
    r = np.array([0,0,0])
    c = np.array([0,2,4])

    data = csr_matrix((v, (r, c)), shape=(2, 6)) 
        
    V = np.array([
      [0.01, 0.02, 0.03, 0.015],
      [0.02, 0.03, 0.05, 0.007],
      [0.05, 0.07, 0.11, 0.101],
      [0.03, 0.01, 0.01, 0.001],
      [0.02, 0.01, 0.09, 0.012],
      [0.05, 0.08, 0.02, 0.002]
    ])
    
    U = np.array([
      [0.02, 0.01, 0.03, 0.04]
    ])
    
    f = looping_update.precompute_f(data, U, V, 0)
    
    assert f.keys() == [0,2,4]
    assert np.allclose(np.array(f.values()), np.array([0.0019, 0.00904, 0.003679]), atol=1e-4)
    
def test_f_optimized():
    v = np.array([0.2,0.3,0.5])
    r = np.array([0,0,0])
    c = np.array([0,2,4])

    data = csr_matrix((v, (r, c)), shape=(2, 6)) 
        
    V = np.array([
      [0.01, 0.02, 0.03, 0.015],
      [0.02, 0.03, 0.05, 0.007],
      [0.05, 0.07, 0.11, 0.101],
      [0.03, 0.01, 0.01, 0.001],
      [0.02, 0.01, 0.09, 0.012],
      [0.05, 0.08, 0.02, 0.002]
    ])
    
    U = np.array([
      [0.02, 0.01, 0.03, 0.04]
    ])
    
    (k,v) = xclimf.precompute_f(data, U, V, 0)
    
    assert np.all(k == [0,2,4])
    assert np.allclose(v, np.array([0.0019, 0.00904, 0.003679]), atol=1e-4)
    
def test_f_performance():
    v = np.array([0.2,0.3,0.5,0.7,0.4,0.2])
    r = np.array([0,0,0,0,0,0])
    c = np.array([0,1,2,3,4,5])

    data = csr_matrix((v, (r, c)), shape=(2, 6)) 
    f = {0: 0.04, 2: 0.07, 3: 0.05, 4: 0.01, 5: 0.02}
    
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
    for i in xrange(5000):
        looping_update.precompute_f(data, U, V, 0)
    time_looping = (datetime.now()-b).total_seconds()
    
    b = datetime.now()
    for i in xrange(5000):
        xclimf.precompute_f(data, U, V, 0)
    time_vector = (datetime.now()-b).total_seconds()

    print "looping", time_looping
    print "vector", time_vector
    assert time_looping > time_vector
