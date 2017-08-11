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
    
def to_matrix(data):
    va = np.array(data["vals"])
    ra = np.array(data["rows"])
    ca = np.array(data["cols"])
    nr = ra.max() + 1
    nc = ca.max() + 1
    return csr_matrix((va, (ra, ca)), shape=(nr, nc))  
    
def put_matrix_data(user, top, data):
    for t in top:
        data["rows"].append(user)
        data["cols"].append(int(t[0]))
        data["vals"].append(t[1])
    
def split_train_test(data, topitems, perctest, topktrain, topktest):
    dtrain = defaultdict(list)
    dtest = defaultdict(list)
    
    users = list(data.iteritems())
    
    for userid in xrange(len(users)):
        user = users[userid][0]
        ratings = users[userid][1]
    
        filtered = filter(lambda a: a[0] not in topitems, ratings)
        sortedit = sorted(filtered, key=lambda a: a[1])
        top = np.array(sortedit[-(topktrain+topktest):])
        np.random.shuffle(top)
        
        toptrain = top[:topktrain]
        put_matrix_data(userid, toptrain, dtrain)
        if random.random() < perctest:
            toptest = top[topktrain:]
            put_matrix_data(userid, toptest, dtest)
    
    train = to_matrix(dtrain)
    test = to_matrix(dtest)

    return (train, test)
    
def split_many_train_test(num, data, topitems, perctest, topktrain, topktest):
    matrixes = []
    for i in xrange(num):
        matrixes.append(split_train_test(data, topitems, perctest, topktrain, topktest))
    return matrixes

