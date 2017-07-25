from collections import defaultdict
import random
import numpy as np
from scipy.sparse import csr_matrix

def read_users_and_items(filename, sep, skip):
    items = defaultdict(float)
    users = defaultdict(list)
    with open(filename, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            if not skip:
                r = line.split(sep)
                m = r[0]
                j = int(r[1])-1
                v = float(r[2])
                users[m].append((j, v))
                items[j] += v
            else:
                skip = False
    return (users, items)
    
def top_items(items):
    return dict(sorted(items.iteritems(), key=lambda a: a[1])[-3:])
    
def split_folds(users, kfolds):
    total = len(users)
    userscopy = users.copy()
    sample_size = total / kfolds
    folds = []
    for i in xrange(kfolds):
        keys = random.sample(list(userscopy), sample_size)
        fold = {}
        for k in keys:
            fold[k] = userscopy[k]
            del userscopy[k]
        folds.append(fold)
    return folds
    
def to_matrix(rows, cols, values):
    va = np.array(values)
    ra = np.array(rows)
    ca = np.array(cols)
    nr = ra.max() + 1
    nc = ca.max() + 1
    return csr_matrix((va, (ra, ca)), shape=(nr, nc))  
    
def split_train_test(fold, topitems, topk):
    rows_train = []
    cols_train = []
    values_train = []
    rows_test = []
    cols_test = []
    values_test = []
    
    users = list(fold.iteritems())
    
    for userid in xrange(len(users)):
        user = users[userid][0]
        ratings = users[userid][1]
    
        filtered = filter(lambda a: a[0] not in topitems, ratings)
        top = np.array(sorted(filtered, key=lambda a: a[1])[-topk*2:])
        np.random.shuffle(top)
        
        tops = np.split(top, 2)
        
        for i in tops[0]:
            rows_train.append(userid)
            cols_train.append(i[0])
            values_train.append(i[1])
        for i in tops[1]:
            rows_test.append(userid)
            cols_test.append(i[0])
            values_test.append(i[1])
            
    train = to_matrix(rows_train, cols_train, values_train)
    test = to_matrix(rows_test, cols_test, values_test)
    
    return (train, test)
    
def split_train_test_many_folds(folds, topitems, topk):
    matrixes = []
    for fold in folds:
        matrixes.append(split_train_test(fold, topitems, topk))
    return matrixes

