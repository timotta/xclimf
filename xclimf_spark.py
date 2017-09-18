from math import exp, log
import numpy as np
import command
import xclimf
import dataset
import pyspark
import random

from pyspark.mllib.common import _py2java, callJavaFunc, _java2py

def quiet_logs( sc ):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
  logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

def alg(train, opts):
    sc = pyspark.SparkContext("local")
    sc.setCheckpointDir("/tmp/" + str(random.random()))
    quiet_logs(sc)

    PyXCLiMF = sc._jvm.com.timotta.rec.xclimf.PyXCLiMF

    vals = []
    for u in xrange(train.shape[0]):
        for i in train[u].indices:
            vals.append( {"user": str(u), "item": str(i), "rating": float(train[u,i])} )

    ratings = sc.parallelize(vals)
    
    x = PyXCLiMF(opts.iters, opts.D, opts.lbda, opts.gamma, opts.topktrain, 0, 1e-4)
    model = x.fit(_py2java(sc, ratings))
    
    U = []
    for ut in _java2py(sc, model.getUserFactors()).sortBy(lambda a: int(a[0])).collect():
        U.append(ut[1])
        
    rddItems = _java2py(sc, model.getItemFactors())
    maxItem = rddItems.map(lambda a: int(a[0])).max()
    items = dict(rddItems.sortBy(lambda a: int(a[0])).collect())
        
    V = []
    for i in xrange(maxItem):
        item = items.get(str(i), np.zeros(opts.D))
        V.append(item)      

    print(V)

    return (U, V)

def main():
    parser = command.options()
    parser.add_option('--dim',dest='D',type='int',default=10,help='dimensionality of factors (default: %default)')
    parser.add_option('--lambda',dest='lbda',type='float',default=0.001,help='regularization constant lambda (default: %default)')
    parser.add_option('--gamma',dest='gamma',type='float',default=0.0001,help='gradient ascent learning rate gamma (default: %default)')
    parser.add_option('--iters',dest='iters',type='int',default=25,help='max iterations (default: %default)')

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
    
    (train, test) = dataset.split_train_test(
      users, topitems, 0.1, 
      opts.topktrain, opts.topktest,
      opts.seltype, opts.norm
    )
    
    (U, V) = alg(train, opts)
    
    print("xclimf spark finished...")
    
    print "train mrr", xclimf.compute_mrr(train, U, V)
    print "train mrr@5", xclimf.compute_mrr(train, U, V, 5)
    print "test mrr", xclimf.compute_mrr(test, U, V)
    print "test mrr@5", xclimf.compute_mrr(test, U, V, 5)

if __name__=='__main__':
    main()

