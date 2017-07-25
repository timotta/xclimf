import csv
import random
from collections import defaultdict
import numpy as np
import xclimf
import dataset
import itertools

import os
import signal
import psutil
from optparse import OptionParser
from multiprocess import Pool, cpu_count

def run(train, test, D, lbda, gamma, eps=0.1):
    print "=" * 80
    print D, lbda, gamma

    U = 0.01*np.random.random_sample((train.shape[0],D))
    V = 0.01*np.random.random_sample((train.shape[1],D))

    last_objective = float("-inf")

    for i in xrange(25):
        xclimf.update(train, U, V, lbda, gamma)
        objective = xclimf.objective(train, U, V, lbda)
        print("interaction %d: %f" % (i,objective) )
        
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
    
def run_safe_many(folds, D, lbda, gamma):
    return [run_safe(f[0], f[1], D, lbda, gamma) for f in folds]
    
def print_better(results):
    print "better until now", sorted(results, key=lambda a: -a[0])
    
parent_id = os.getpid()
def worker_init():
    def sig_int(signal_num, frame):
        print('signal: %s' % signal_num)
        parent = psutil.Process(parent_id)
        for child in parent.children():
            if child.pid != os.getpid():
                print("killing child: %s" % child.pid)
                child.kill()
        print("killing parent: %s" % parent_id)
        parent.kill()
        print("suicide: %s" % os.getpid())
        psutil.Process(os.getpid()).kill()
    signal.signal(signal.SIGINT, sig_int)
    
def grid_search(folds, cores):
    Ds = [10,15,20,25]
    lbdas = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    
    pool = Pool(cores, worker_init)

    results = []

    jobresults = []
    for p in itertools.product(Ds, lbdas, gammas):
        jobresults.append( pool.apply_async(run_safe_many, (folds, p[0], p[1], p[2])) )
    for r in jobresults:
        mmrr = np.mean(r.get())
        results.append((mmrr, p))
        print_better(results)
        
    return results

def main():
    parser = OptionParser()
    parser.add_option('--dataset',dest='dataset', help='training dataset')
    parser.add_option('--sep',dest='sep',default="\t",help='string used to split colums in dataset  (default: %default)')
    parser.add_option('--topk',dest='topk',type=int,default=5,help='number of topk items used for each user  (default: %default)')
    parser.add_option('--kfolds',dest='kfolds',type=int,default=3,help='number of folds used for cross-validation  (default: %default)')
    parser.add_option('--cores',dest='cores',type=int,default=cpu_count()-1,help='number of cores to run in parallel (default: %default)')
    parser.add_option('--skipfl',dest='skipfl',action="store_true",help='should skip dataset first line or not (default: %default)')
    
    (opts,args) = parser.parse_args()
    if not opts.dataset:
        parser.print_help()
        raise SystemExit
    
    print("reading %s..." % opts.dataset)
    
    (users, items) = dataset.read_users_and_items(opts.dataset, opts.sep, opts.skipfl)
    
    print("loaded %d users" % len(users))
    print("loaded %d items" % len(items))
    
    topitems = dataset.top_items(items)
    
    print("do not use these top items %s" % str(topitems))
    
    folds = dataset.split_folds(users, opts.kfolds)
    
    print("splited into %d folds of %d users" % (len(folds), len(folds[0])) )
    
    folds_matrixes = dataset.split_train_test_many_folds(folds, topitems, opts.topk)
    
    results = grid_search(folds_matrixes, opts.cores)
    
    print("="*80)
    print_better(results)

if __name__=='__main__':
    main()
    
    
