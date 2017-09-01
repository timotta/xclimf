from scipy.sparse import csr_matrix
import xclimf
import numpy as np
from datetime import datetime
import looping_update

def test_objective_1():
    v = np.array([0.2,0.5])
    r = np.array([0,0])
    c = np.array([0,2])

    data = csr_matrix((v, (r, c)), shape=(2, 6)) 
    
    V = np.array([
      [0.01, 0.02, 0.03, 0.15],
      [0.02, 0.03, 0.05, 0.07],
      [0.05, 0.07, 0.11, 0.101]
    ])
    
    U = np.array([[0.02, 0.01, 0.03, 0.04]])
    
    objV = xclimf.objective(data, U, V, 0.01)
    objL = looping_update.objective(data, U, V, 0.01)
   
    assert abs(objV - objL) < 1e-12
    
def test_objective_2():
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
    
    objV = xclimf.objective(data, U, V, 0.01)
    objL = looping_update.objective(data, U, V, 0.01)
   
    assert abs(objV - objL) < 1e-12
    
def test_objective_performance():
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
        xclimf.objective(data, U, V, 0.01)
    time_vector = (datetime.now()-b).total_seconds()
    
    b = datetime.now()
    for i in xrange(1000):
        looping_update.objective(data, U, V, 0.01)
    time_looping = (datetime.now()-b).total_seconds()
    
    print "looping", time_looping
    print "vector", time_vector
    assert time_looping > time_vector
