#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
File to clean up text files
'''

import sys, os

f = open(sys.argv[1], 'r').readlines()

fw = open(sys.argv[1]+'_norm', 'w')

for line in f:
    if line == '\n':
        continue
    key, words = line.split(' ', 1)
    words = words.lower()
    words = words.replace('[sounds like]', '[sounds-like]')
    words = words.replace('[start song]', '[start-song]')
    words = words.replace('[end song]', '[end-song]')
    words = words.replace('[plays ', '[plays-music]')
    fw.write(key+' '+words)
fw.close()

a = os.system('mv '+sys.argv[1]+' '+ sys.argv[1]+'_orig')
b = os.system('mv '+sys.argv[1]+'_norm '+ sys.argv[1])
assert a == 0
assert b == 0
