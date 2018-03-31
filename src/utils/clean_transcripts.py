'''
File to clean up text files
'''

import sys, re

f = open(sys.argv[1], 'r').readlines()

fw = open(sys.argv[1]+'_norm', 'w')

for line in f:
    line = line.strip()
    line = line.lower()
    regex = re.compile(r"\#\$\%\&\*\+\=\@_\[\]  ^.*interfaceOpDataFile.*$", re.IGNORECASE)

