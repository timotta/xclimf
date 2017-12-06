from collections import defaultdict
import random
import numpy as np
from scipy.sparse import csr_matrix

def read_users_and_items(filename, sep, skip):
    usersmap = {}
    itemsmap = {}
    items = defaultdict(float)
    users = defaultdict(list)
    
    mi = 0
    ji = 0
    
    with open(filename, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            if not skip:
                r = line.split(sep)
                
                if r[0] not in usersmap:
                    usersmap[r[0]] = mi
                    mi = mi + 1
                m = usersmap[r[0]]
                
                if r[1] not in itemsmap:
                    itemsmap[r[1]] = ji
                    ji = ji + 1
                j = itemsmap[r[1]]
                    
                v = float(r[2])
                users[str(m)].append((j, v))
                items[j] += v
                m = m + 1
            else:
                skip = False
    return (users, items)
    
def top_items(items, topK=3):
    if topK == 0:
        return {}
    return dict(sorted(items.iteritems(), key=lambda a: a[1])[-topK:])

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
    
def split_train_test(data, topitems, perctest, ktrain, ktest, seltype="top", norm=False):
    dtrain = defaultdict(list)
    dtest = defaultdict(list)
    
    users = list(data.iteritems())
    
    for userid in xrange(len(users)):
        user = users[userid][0]
        ratings = users[userid][1]
    
        filtered = filter(lambda a: a[0] not in topitems, ratings)
        
        if seltype == "random":
            top = np.array(filtered)
            np.random.shuffle(top)
            top = top[-(ktrain+ktest):]
        else:
            sortedit = sorted(filtered, key=lambda a: a[1])
            top = np.array(sortedit[-(ktrain+ktest):])
            np.random.shuffle(top)

        if len(top) > ktrain:
            cut = ktrain
        elif len(top) > 2:
            cut = len(top) - 1
        else:
            cut = len(top)
                
        toptrain = top[:cut]
        put_matrix_data(userid, toptrain, dtrain)
        if len(top) > 2 and random.random() < perctest:
            toptest = top[cut:]
            put_matrix_data(userid, toptest, dtest)
    
    train = to_matrix(dtrain)
    test = to_matrix(dtest)

    if norm:
        m = max(train.max(), test.max())
        train = train/m
        test = test/m

    return (train, test)
    
def split_many_train_test(num, data, topitems, perctest, topktrain, topktest, seltype, norm):
    matrixes = []
    for i in xrange(num):
        matrixes.append(split_train_test(
          data, topitems, perctest, topktrain, topktest, seltype, norm
        ))
    return matrixes

