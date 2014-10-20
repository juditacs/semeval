from sys import stdin
from collections import defaultdict
from argparse import ArgumentParser


def parse_args():
    p = ArgumentParser('Reads paraphrase candidates from STDIN. Format: phrase1 <TAB> phrase2 [<TAB> gold]')
    p.add_argument('-l', '--lower', action='store_true', default=False,
                   help='lower all input')
    p.add_argument('--mode', choices=['word', 'char'], default='char',
                   help='word or character ngrams')
    p.add_argument('-N', type=int, default=3, help='N of the Ngram')
    p.add_argument('-t', '--threshold', type=float, default=0.5,
                   help='Jaccard threshold')
    p.add_argument('--silent', action='store_true', default=False,
                   help='do not print the Jaccard score for each pair, '
                   'print only the summary.')
    p.add_argument('--roc', type=str, default='',
                   help='produce a ROC table')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='print verbose summary (true/false positives and negatives)')
    p.add_argument('--strict', type=int, default=1,
                   help='good - bad = x, accept x>=strict answers. E.g.'
                   'if strict is 2, then 4-1=2 is accepted, 3-2=1 is not accepted.'
                   'This argument has no effect if there is no third column in the input')
    p.add_argument('--round-to', type=int, default=3,
                   help='round prec/rec/acc/F values to R')
    return p.parse_args()

cfg = parse_args()
roc_thresholds = [float(i) / 100 for i in xrange(0, 101, 1)]


def trim_text(text):
    if cfg.lower:
        return text.lower()
    return text


def get_ngrams(text):
    ngrams = set()
    trimmed = trim_text(text)
    N = cfg.N
    if cfg.mode == 'char':
        for i in xrange(0, len(trimmed) - N + 1):
            ngrams.add(trimmed[i:i + N])
    else:
        words = trimmed.split(' ')
        for i in xrange(0, len(words) - N + 1):
            ngrams.add(tuple(words[i:i + N]))
    return ngrams


def get_answer(field):
    good = int(field[1])
    bad = int(field[-2])
    return good - bad >= cfg.strict


def jaccard(set1, set2):
    return float(len(set1 & set2)) / len(set1 | set2)


def store_stat(stat, jacc, gold):
    if cfg.roc:
        for th in roc_thresholds:
            store_with_threshold(jacc, gold, th, stat[th])
    else:
        store_with_threshold(jacc, gold, cfg.threshold, stat)


def store_with_threshold(jacc, gold, th, stat):
    if jacc >= th:
        guess = True
    else:
        guess = False
    if gold and guess:
        stat['tp'] += 1
    elif gold and not guess:
        stat['fn'] += 1
    elif not gold and guess:
        stat['fp'] += 1
    else:
        stat['tn'] += 1
    stat['N'] += 1


def get_measures(stat):
    acc = float(stat['tp'] + stat['tn']) / stat['N']
    try:
        prec = float(stat['tp']) / (stat['tp'] + stat['fp'])
    except ZeroDivisionError:
        prec = 0
    try:
        rec = float(stat['tp']) / (stat['tp'] + stat['fn'])
    except ZeroDivisionError:
        rec = 0
    try:
        F = 2 * prec * rec / (prec + rec)
    except ZeroDivisionError:
        F = 0
    return acc, prec, rec, F


def compute_and_print_stats(stat):
    if cfg.verbose:
        print('true pos: {0}, true neg: {1}, false pos: {2}, false neg: {3}'.format(
            stat['tp'], stat['tn'], stat['fp'], stat['fn']))
    acc, prec, rec, F = get_measures(stat)
    r = cfg.round_to
    print('| Accuracy | Precision | Recall | F score |\n| --- | --- | --- | --- |')
    print('| {0} | {1} | {2} | {3} |'.format(round(acc, r), round(prec, r), round(rec, r), round(F, r)))


def main():
    if cfg.roc:
        stat = defaultdict(lambda: defaultdict(int))
    else:
        stat = defaultdict(int)
    for l in stdin:
        fd = l.decode('utf8').strip().split('\t')
        feat1 = get_ngrams(fd[0])
        feat2 = get_ngrams(fd[1])
        jacc = jaccard(feat1, feat2)
        if not cfg.silent:
            print(u'{0}\t{1}\t{2}'.format(jacc, fd[0], fd[1]).encode('utf8'))
        if len(fd) > 2:
            answer = get_answer(fd[2])
            store_stat(stat, jacc, answer)
    if stat and not cfg.roc:
        compute_and_print_stats(stat)
    if cfg.roc:
        with open(cfg.roc, 'w') as f:
            for th, st in sorted(stat.iteritems(), key=lambda x: -x[0]):
                acc, prec, rec, F = get_measures(st)
                N = float(st['N'])
                tp_rate = st['tp'] / N
                fp_rate = st['fp'] / N
                f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(th, tp_rate, fp_rate,
                        acc, prec, rec, F))


if __name__ == '__main__':
    main()
