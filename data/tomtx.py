import csv

rows = []
cols = []
values = []

with open('ml-100k/u1.test', 'rb') as csvfile:
    c = csv.reader(csvfile, delimiter='\t')
    for r in c:
        rows.append(int(r[0])-1)
        cols.append(int(r[1])-1)
        values.append(float(r[2]))



from scipy.sparse import csr_matrix
import numpy as np

va = np.array(values)
ra = np.array(rows)
ca = np.array(cols)

vm = va.max()
nr = ra.max() + 1
nc = ca.max() + 1

va = va/vm

mtx = csr_matrix((va, (ra, ca)), shape=(nr, nc))
print mtx.toarray()

from scipy.io.mmio import mmwrite, mmread
mmwrite("u1_test.mtx", mtx)

mtxr = mmread("u1_test.mtx").tocsr()
print mtxr.toarray()
