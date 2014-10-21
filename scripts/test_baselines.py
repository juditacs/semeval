import sys

def dice(e1, e2, ei):
    return 2 * ei / (e1 + e2)

def jaccard(e1, e2, ei):
    return ei / (e1 + e2 - ei)

def int_per_smaller(e1, e2, ei):
    if e1 == 0 or e2 == 0:
        return 0
    return ei / min(e1, e2)

def main():
    tst_gs = open('test.gs', 'w')
    tst_out = open('test.out', 'w')
    include, exclude = None, None
    if len(sys.argv) > 2:
        include = set(sys.argv[2:])
    for line in sys.stdin:
        fields = line.strip().split()
        batch = fields[1].split('.')[0]
        if ((include and batch not in include) or
                (exclude and batch in exclude)):
            continue
        _, gold, e1, e2, ei = [
            float(fields[i].strip(',')) for i in (3, 5, 7, 9, 11)]

        sim = 5 * __BASELINE__(e1, e2, ei)
        tst_gs.write("{0}\n".format(gold))
        tst_out.write("{0}\n".format(sim))

__BASELINE__ = globals()[sys.argv[1]]

if __name__ == '__main__':
    main()
