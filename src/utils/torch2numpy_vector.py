

import sys, os, pdb
import numpy as np
import cPickle

a = np.zeros((324,32), dtype=float)

with open(sys.argv[1], 'r') as f2:
    print f2
    x = cPickle.load(f2)
    print x.size()
    pdb.set_trace()
    a = x.data.numpy()[0]

np.savetxt(sys.argv[1]+'_numpy.txt', a)
    

