'''

'''

import sys

trn = open(sys.argv[1], 'r').readlines()
txt = open(sys.argv[1].split('.', 1)[0]+'.txt', 'w')

for line in trn:
	line = line.strip()
	if line == '':
		continue
	words = line.split(' ')
	key = words[-1]
	key = key.strip('(').strip(')')
	key = key.split('-', 1)[1]
	words = words[:-1]
	wrd_str = " ".join(words)
	txt.write(key+' '+wrd_str+'\n')
txt.close()

