#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
File to make word dictionary, and
File to clear word dictionary of false starts, hesitations, split hyphenated words into 2, preserve words in [...]
'''

import sys
from collections import Counter

text = open(sys.argv[1], 'r').readlines()

# get unique words, handle special symbols within [..]
cnt = Counter()
for line in text:
    line = line.strip()
    if line == '':
        continue
    curr_words = line.split(' ')[1:]
    for w in curr_words:
        if w == '':
            continue
        cnt[w] += 1

words = []
for w in cnt.keys():
    if cnt[w] >=5:
        words.append(w)

# write to file
d = open(sys.argv[2], 'w')

d.write('<unk> 1\n')
idx = 2
for w in words:
    d.write(w +' '+str(idx)+'\n')
    idx += 1
d.close()
