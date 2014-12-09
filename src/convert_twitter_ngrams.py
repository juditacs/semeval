""" convert twitter ngram corpus to word2vec format
"""
from sys import argv, stdin, stderr
import re
import math


def main():
    cutoff = int(argv[1])
    print_re = re.compile(r'[^[:print:]]', re.UNICODE)
    line_cnt = 0
    for l in stdin:
        line_cnt += 1
        if line_cnt % 1000000 == 0:
            stderr.write(str(line_cnt) + '\n')
        fd = l.decode('utf8').strip().split('\t')
        ngram = fd[0]
        if print_re.search(ngram):
            continue
        cnt = sum(int(i) for i in fd[1:])
        if cnt >= cutoff:
            for i in xrange(int(math.log(cnt)) + 1):
                print(ngram.encode('utf8', 'ignore'))


if __name__ == '__main__':
    main()
