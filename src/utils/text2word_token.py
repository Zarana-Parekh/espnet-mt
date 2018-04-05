#!/usr/bin/env python2

'''
Helper file to convert text to word-tokens
Skips the non langauage words
'''
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-ncols', '-s', default=0, type=int,
                        help='skip first n columns')
    parser.add_argument('text', type=str, default=False, nargs='?',
                        help='input text')
    parser.add_argument('--non-lang-syms', '-l', default=None, type=str,
                        help='list of non-linguistic symobles, e.g., <NOISE> etc.')
    args = parser.parse_args()

    if args.non_lang_syms is not None:
        with open(args.non_lang_syms, 'r') as f:
            nls = [unicode(x.rstrip(), 'utf-8') for x in f.readlines()]

    if args.text:
        f = open(args.text)
    else:
        f = sys.stdin

    line = f.readline()

    while line:
        x = unicode(line, 'utf_8').split()
        print ' '.join(x[:args.skip_ncols]).encode('utf_8'),
        a = ' '.join(x[args.skip_ncols:])

	#a = [a[i:i + len(a)] for i in range(0, len(a), len(a))]
	a = a.split()

        a_flat = []
        for z in a:
            #if z in nls:
                # skipping nlsyms
            #    continue
            a_flat.append("".join(z))
        print ' '.join(a_flat).encode('utf_8')
        line = f.readline()


if __name__ == '__main__':
    main()
