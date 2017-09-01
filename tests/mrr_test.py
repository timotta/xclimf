from scipy.sparse import csr_matrix
import numpy as np

from xclimf import compute_mrr

def test_compute_mrr():
  row = np.array([0,   0,   1,   1])
  col = np.array([0,   1,   0,   1])
  val = np.array([1.0, 2.0, 2.0, 1.0])
  data = csr_matrix((val, (row, col)), shape=(2, 2))
  
  U = np.array([
    [1.0, 2.0],
    [2.0, 1.0]
  ])
  V = np.array([
    [2.0, 1.0],
    [1.0, 2.0]
  ])
  
  mrr = compute_mrr(data, U, V)

  assert mrr == 1
  
def test_compute_mrr_when_dont_predicted_for_user():
  row = np.array([0,   0,   1])
  col = np.array([0,   1,   2])
  val = np.array([1.0, 2.0, 2.0])
  data = csr_matrix((val, (row, col)), shape=(2, 3))
  
  U = np.array([
    [1.0, 2.0],
    [2.0, 1.0]
  ])
  V = np.array([
    [2.0, 1.0],
    [1.0, 2.0],
    [0.0, 0.0],
  ])
  
  mrr = compute_mrr(data, U, V)

  assert mrr == (1.0 + 1.0/3.0)/2.0
  
