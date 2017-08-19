from optparse import OptionParser

def options():
    parser = OptionParser()
    parser.add_option('--dataset',dest='dataset', help='path to the dataset file')
    parser.add_option('--sep',dest='sep',default="\t",help='string used to split colums in dataset')
    parser.add_option('--skipfl',dest='skipfl',action="store_true",help='should skip dataset first line or not (default: %default)')
    parser.add_option('--topktrain',dest='topktrain',type=int,default=5,help='number of topk training items for each user (default: %default)')
    parser.add_option('--topktest',dest='topktest',type=int,default=5,help='number of topk test items for each user (default: %default)')
    parser.add_option('--kfolds',dest='kfolds',type=int,default=3,help='number of folds for cross-validation (default: %default)')
    parser.add_option('--seltype',dest='seltype',default="top",help='rating selection type by user [random|top] (default: %default)')
    return parser
    
 
