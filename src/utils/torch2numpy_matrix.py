

import sys, os, pdb
import numpy as np
import cPickle

folder = sys.argv[1]

filenames = os.listdir(folder)

print 'total files = ', len(filenames)

a = np.zeros((len(filenames),159), dtype=float)

count = 0
for f in filenames:
    with open((sys.argv[1]+'/'+f), 'r') as f2:
        print f
        x = cPickle.load(f2)
        print x.size()
        a[count] = x.data.numpy()[0]
        count += 1

np.savetxt(sys.argv[1]+'/2numpy.txt', a)
    

