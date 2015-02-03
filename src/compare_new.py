from sys import argv


def read_scores(fn):
    with open(fn) as f:
        return [float(i) for i in f]


def read_sentences(fn):
    sent = []
    with open(fn) as f:
        for l in f:
            fs = l.decode('utf8').strip().split('\t')
            if len(fs) < 3:
                sent.append((fs[0], fs[1]))
            else:
                sent.append((fs[2], fs[3]))
    return sent


def main():
    old_f = argv[1]
    new_f = argv[2]
    sen_f = argv[3]
    old_scores = read_scores(old_f)
    new_scores = read_scores(new_f)
    sent = read_sentences(sen_f)
    for i, old_sc in enumerate(old_scores):
        if not old_sc == new_scores[i]:
            print(u'{0}\t{1}\t{2}\t{3}\t{4}'.format(i + 1, old_sc, new_scores[i], sent[i][0], sent[i][1]).encode('utf-8'))

if __name__ == '__main__':
    main()
