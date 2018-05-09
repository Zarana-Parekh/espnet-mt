import sys

fname = open(sys.argv[1], 'r')

for line in fname:
	line = line.strip()[line.find(']: ') + 3:]
	print line

fname.close()


