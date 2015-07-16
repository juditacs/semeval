#!/usr/bin/env python
"""read a TAB-delimited file and add a column containing the difference of the
last two columns"""

import sys

def main():
    for line in sys.stdin:
        x, y = map(float, line.strip().split('\t')[-2:])
        x /= 5
        print "{0}\t{1}".format(line.strip(), x - y)

if __name__ == "__main__":
    main()
