import sys

def dice(e1, e2, ei):
    return 2 * ei / (e1 + e2)

def jaccard(e1, e2, ei):
    return ei / (e1 + e2 - ei)

def boosted_jaccard(e1, e2, ei):
    return ei * jaccard(e1, e2, ei)

def int_per_smaller(e1, e2, ei):
    if e1 == 0 or e2 == 0:
        return 0
    return ei / min(e1, e2)

def main():
    tst_gs = open('test.gs', 'w')
    tst_out = open('test.out', 'w')
    for line in sys.stdin:
        fields = line.strip().split()
        try:
            _, gold, e1, e2, ei = [
                float(fields[i].strip(',')) for i in (1, 3, 5, 7, 9)]
        except ValueError:
            raise Exception('invalid line: {0}'.format(fields))

        sim = 5 * __BASELINE__(e1, e2, ei)
        tst_gs.write("{0}\n".format(gold))
        tst_out.write("{0}\n".format(sim))
        print "{0}\t{1}".format(gold, sim)

__BASELINE__ = globals()[sys.argv[1]]

if __name__ == '__main__':
    main()
