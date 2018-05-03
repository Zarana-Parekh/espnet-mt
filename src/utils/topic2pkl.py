'''
File to create one hot representation of topic vectors and dump as pickle

input:
    input utt2topic file

topics in input indexed from 0-21

output:
    topic_feats.p
'''

import sys
import numpy as np
import cPickle as pickle

# input is train, val and test utt2topic file combined into one file
inp = open(sys.argv[1], 'r').readlines()

di = {}
di_onehot = {}

for line in inp:
    line = line.strip()
    if line == '':
        print 'WARNING: empty line'
        continue
    key, val = line.split(' ')
    hot = np.zeros(22, dtype=int)
    hot[int(val)] = 1
    di_onehot[key] = hot

print 'total number of utts (train+val+test): ', len(di_onehot)

# dump dict to pickle
with open('data/visfeats/topic_feats.p', 'wb') as handle:
    pickle.dump(di_onehot, handle, protocol=pickle.HIGHEST_PROTOCOL)


