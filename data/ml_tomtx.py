import csv
from collections import defaultdict
from scipy.sparse import csr_matrix
import numpy as np
from scipy.io.mmio import mmwrite, mmread

INPUT = 'ml-100k/u.data'
OUTPUT_TRAIN = 'ml_upl5_train.mtx'
OUTPUT_TEST = 'ml_upl5_test.mtx'
UPL = 5

items = defaultdict(float)
users = defaultdict(list)
with open(INPUT, 'rb') as csvfile:
    c = csv.reader(csvfile, delimiter='\t')
    for r in c:
        m = int(r[0])-1
        j = int(r[1])-1
        v = float(r[2])
        users[m].append((j, v))
        items[j] += v
        
print "loaded %d users" % len(users)
print "loaded %d items" % len(items)
        
topitens = dict(sorted(items.iteritems(), key=lambda a: a[1])[-3:])

print "do not use these top items", topitens

rows_train = []
cols_train = []
values_train = []
rows_test = []
cols_test = []
values_test = []
for user, ratings in users.iteritems():
    filtered = filter(lambda a: a[0] not in topitens, ratings)
    top = np.array(sorted(filtered, key=lambda a: a[1])[-UPL*2:])
    np.random.shuffle(top)
    tops = np.split(top, 2)
    for i in tops[0]:
        rows_train.append(user)
        cols_train.append(i[0])
        values_train.append(i[1])
    for i in tops[1]:
        rows_test.append(user)
        cols_test.append(i[0])
        values_test.append(i[1])

def save(output, rows, cols, values):
  print "saving ", output
  va = np.array(values)
  ra = np.array(rows)
  ca = np.array(cols)

  nr = ra.max() + 1
  nc = ca.max() + 1

  print "num rows", nr
  print "num columns", nc

  mtx = csr_matrix((va, (ra, ca)), shape=(nr, nc))
  print "writing matrix with shape", mtx.shape
  mmwrite(output, mtx)

  mtxr = mmread(output).tocsr()
  print "wrote matrix with shape", mtxr.shape
  
save(OUTPUT_TRAIN, rows_train, cols_train, values_train)
save(OUTPUT_TEST, rows_test, cols_test, values_test)
