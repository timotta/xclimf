import csv
import random
from collections import defaultdict
import numpy as np
import xclimf
import dataset
import itertools
import command

import os
import signal
import psutil
from optparse import OptionParser
from multiprocess import Pool, cpu_count

def print_interaction(i, objective, U, V, params):
    print("iteraction %d: %f (%s)" % (i,objective,params) )

def run(train, test, params, eps=0.1):
    print "=" * 80
    print params
    
    (U, V) = xclimf.gradient_ascent(train, test, params, foreach=print_interaction)

    trainmrr = xclimf.compute_mrr(train, U, V)
    testmrr = xclimf.compute_mrr(test, U, V)
    print "train mrr", trainmrr
    print "test mrr", testmrr
    return testmrr

def run_safe_many(folds, params):
    try:
        return [run(f[0], f[1], params) for f in folds]
    except Exception, e:
        print(e)
        return -1.0
    
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
        params = {"dims": p[0], "lambda": p[1], "gamma": p[2]}
        jr = (params.copy(), pool.apply_async(run_safe_many, (folds, params)))
        jobresults.append(jr)
    for jr in jobresults:
        mmrr = np.mean(jr[1].get())
        results.append((mmrr, jr[0]))
        print_better(results)
    return results

def main():
    parser = command.options()
    parser.add_option('--cores',dest='cores',type=int,default=cpu_count()-1,help='number of cores to run in parallel (default: %default)')

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
    
    folds = dataset.split_many_train_test(
      opts.kfolds, users, topitems,
      0.1, opts.topktrain, opts.topktest,
      opt.seltype, opts.norm
    )
    
    print("splited into %d folds" % (len(folds)) )

    results = grid_search(folds, opts.cores)
    
    print("="*80)
    print_better(results)

if __name__=='__main__':
    main()
    
    
