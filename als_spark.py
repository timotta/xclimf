from math import exp, log
import numpy as np
import command
import xclimf
import dataset
import pyspark
import random
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

def quiet_logs( sc ):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel( logger.Level.WARN )
  logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

def als(train, opts):
    vals = []
    for u in xrange(train.shape[0]):
        for i in train[u].indices:
            vals.append(Rating(int(u), int(i), float(train[u,i])))
            
    sc = pyspark.SparkContext("local")
    sc.setCheckpointDir("/tmp/" + str(random.random()))
    quiet_logs(sc)
    ratings = sc.parallelize(vals)
    
    if opts.implicit:
        model = ALS.trainImplicit(ratings, opts.D, opts.iters, opts.lbda, alpha=opts.alpha)
    else:
        model = ALS.train(ratings, opts.D, opts.iters, opts.lbda)
    
    U = []
    for ut in model.userFeatures().sortBy(lambda a: a[0]).collect():
        U.append(ut[1])

    rddItems = model.productFeatures()
    maxItem = rddItems.map(lambda a: int(a[0])).max()
    items = dict(rddItems.sortBy(lambda a: int(a[0])).collect())
        
    V = []
    for i in xrange(maxItem):
        item = items.get(i, np.zeros(opts.D))
        V.append(item)

    return (U, V)

def main():
    parser = command.options()
    parser.add_option('--dim',dest='D',type='int',default=10,help='dimensionality of factors (default: %default)')
    parser.add_option('--lambda',dest='lbda',type='float',default=0.001,help='regularization constant lambda (default: %default)')
    parser.add_option('--iters',dest='iters',type='int',default=25,help='max iterations (default: %default)')
    parser.add_option('--ignore',dest='ignore',type='int',default=3,help='ignore top k items (default: %default)')
    parser.add_option('--implicit',dest='implicit',action="store_true",help='implicit feedback')
    parser.add_option('--alpha',dest='alpha',type='float',default=0.01,help='constant to compute confindence (only for implicit) (default: %default)')
    
    (opts,args) = parser.parse_args()
    if not opts.dataset:
        parser.print_help()
        raise SystemExit
        
    (users, items) = dataset.read_users_and_items(opts.dataset, opts.sep, opts.skipfl)
    
    print("loaded %d users" % len(users))
    print("loaded %d items" % len(items))
    
    topitems = dataset.top_items(items, opts.ignore)
    
    print("do not use these top items %s" % str(topitems))
    
    (train, test) = dataset.split_train_test(
      users, topitems, 0.1, 
      opts.topktrain, opts.topktest,
      opts.seltype, opts.norm
    )
    
    (U, V) = als(train, opts) 
    
    print("als finished...")
    
    print "train mrr", xclimf.compute_mrr(train, U, V)
    print "train mrr@5", xclimf.compute_mrr(train, U, V, 5)
    print "test mrr", xclimf.compute_mrr(test, U, V)
    print "test mrr@5", xclimf.compute_mrr(test, U, V, 5)
  
if __name__=='__main__':
    main()
