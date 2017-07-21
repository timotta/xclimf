import csv
from scipy.sparse import csr_matrix
from collections import defaultdict
import numpy as np
import xclimf
import itertools

def read_users_and_items(filename, sep):
    items = defaultdict(float)
    users = defaultdict(list)
    with open(filename, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            r = line.split(sep)
            m = r[0]
            j = int(r[1])-1
            v = float(r[2])
            users[m].append((j, v))
            items[j] += v
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

def split_train(fold, topitems, topk):
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
        matrixes.append(split_train(fold, topitems, topk))
    return matrixes
    
def run(train, test, D, lbda, gamma, eps=0.1):
    print "=" * 80
    print D, lbda, gamma

    U = 0.01*np.random.random_sample((train.shape[0],D))
    V = 0.01*np.random.random_sample((train.shape[1],D))

    last_objective = float("-inf")

    for i in xrange(25):
        xclimf.update(train, U, V, lbda, gamma)
        objective = xclimf.objective(train, U, V, lbda)
        print "objective: ", objective
        
        if objective > last_objective:
            last_objective = objective
        elif objective < last_objective + eps:
            print "objective should be bigger or equal last objective..."
            break
    
    trainmrr = xclimf.compute_mrr(train, U, V)
    testmrr = xclimf.compute_mrr(test, U, V)
    print "train mrr", trainmrr
    print "test mrr", testmrr
    return testmrr

def run_safe(train, test, D, lbda, gamma):
    try:
        return run(train, test, D, lbda, gamma)
    except Exception, e:
        print(e)
        return 0.0
    
def print_better(results):
    print "better until now", sorted(results, key=lambda a: -a[0])
    
def validate(folds):
    Ds = [10,15,20,25]
    lbdas = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    results = []
    for p in itertools.product(Ds, lbdas, gammas):
        mrrs = [run_safe(f[0], f[1], p[0], p[1], p[2]) for f in folds]
        mmrr = np.mean(mrrs)
        results.append((mmrr, p))
        print_better(results)
    return results

if __name__=='__main__':

    from optparse import OptionParser
    import random

    parser = OptionParser()
    parser.add_option('--dataset',dest='dataset',help='training dataset')
    parser.add_option('--sep',dest='sep',default="\t",help='string used to split colums in dataset')
    parser.add_option('--topk',dest='topk',type=int,default=5,help='number of topk items used for each user')
    parser.add_option('--kfolds',dest='kfolds',type=int,default=3,help='number of folds used for cross-validation')
    
    (opts,args) = parser.parse_args()
    if not opts.dataset:
        parser.print_help()
        raise SystemExit
    
    (users, items) = read_users_and_items(opts.dataset, opts.sep)
    
    print("loaded %d users" % len(users))
    print("loaded %d items" % len(items))
    
    topitems = top_items(items)
    
    print("do not use these top items", topitems)
    
    folds = split_folds(users, opts.kfolds)
    
    print("splited into", len(folds), "folds of", len(folds[0]), "users")
    
    folds_matrixes = split_train_test_many_folds(folds, topitems, opts.topk)
    
    results = validate(folds_matrixes)
    
    print("="*80)
    print_better(results)
    
    
