import numpy as np
from scipy.sparse import csr_matrix
from scipy.io.mmio import mmread

epinions = mmread("data/EP25_UPL5_train.mtx").tocsr()
ml = mmread("data/u1_train.mtx").tocsr()
mlupl = mmread("data/ml_upl5_train.mtx").tocsr()

def mean_len(data):
  us = data.shape[0]
  return sum([len(data[m].indices) for m in xrange(us)]) / float(us)
  
print("epinions", mean_len(epinions))
print("ml", mean_len(ml))
print("ml_upl", mean_len(mlupl))
    
