from math import exp, log
import numpy as np
import command
import xclimf
import dataset
import pyspark
import random
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

def als(train, opts):
    vals = []
    for u in xrange(train.shape[0]):
        for i in train[u].indices:
            vals.append(Rating(int(u), int(i), float(train[u,i])))
            
    sc = pyspark.SparkContext("local")
    sc.setCheckpointDir("/tmp/" + str(random.random()))
    ratings = sc.parallelize(vals)
    
    model = ALS.train(ratings, opts.D, opts.iters, opts.lbda)
    
    U = []
    for ut in model.userFeatures().sortBy(lambda a: a[0]).collect():
        U.append(ut[1][1])

    V = []
    for vt in model.productFeatures().sortBy(lambda a: a[0]).collect():
        V.append(vt[1][1])

    return (U, V)

def main():
    parser = command.options()
    parser.add_option('--dim',dest='D',type='int',default=10,help='dimensionality of factors (default: %default)')
    parser.add_option('--lambda',dest='lbda',type='float',default=0.001,help='regularization constant lambda (default: %default)')
    parser.add_option('--iters',dest='iters',type='int',default=25,help='max iterations (default: %default)')
    
    (opts,args) = parser.parse_args()
    if not opts.dataset:
        parser.print_help()
        raise SystemExit
        
    (users, items) = dataset.read_users_and_items(opts.dataset, opts.sep, opts.skipfl)
    
    print("loaded %d users" % len(users))
    print("loaded %d items" % len(items))
    
    topitems = dataset.top_items(items)
    
    print("do not use these top items %s" % str(topitems))
    
    (train, test) = dataset.split_train_test(
      users, topitems, 0.1, 
      opts.topktrain, opts.topktest,
      opts.seltype, opts.norm
    )
    
    (U, V) = als(train, opts) 
    
    print("als finished...")
    
    trainmrr = xclimf.compute_mrr(train, U, V)
    testmrr = xclimf.compute_mrr(test, U, V)
    print "train mrr", trainmrr
    print "test mrr", testmrr
  
if __name__=='__main__':
    main()
