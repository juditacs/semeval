from numpy import linalg
from sentence import SentencePair
from collections import defaultdict


class Regression(object):

    def __init__(self, conf, reader, aligner):
        self.conf = conf
        self.reader = reader
        self.aligner = aligner
        self._feat_order = {}
        self._feat_i = 0

    def regression(self):
        with open(self.conf.get('regression', 'train')) as f:
            self.train = self.featurize(f)
        with open(self.conf.get('regression', 'train_labels')) as f:
            self.train_labels = self.read_labels(f)
        self.reader.clear_pairs()
        with open(self.conf.get('regression', 'test')) as f:
            self.test = self.featurize(f)
        self.train_feats = self.convert_to_table(self.train)
        self.test_feats = self.convert_to_table(self.test)
        self.model = linalg.lstsq(self.train_feats, self.train_labels)[0]
        self.predicted = self.predict_regression(self.test_feats, 0.3)
        with open(self.conf.get('regression', 'outfile'), 'w') as f:
            f.write('\n'.join(str(int(i if i < 0.5 else 1.0)) for i in self.predicted) + '\n')

    def read_labels(self, stream, true_th=0.5):
        labels = []
        for l in stream:
            f = float(l.strip().split('\t')[-1])
            if self.conf.getboolean('regression', 'binary_labels'):
                f = 0 if f < true_th else 1
            labels.append(f)
        return labels

    def featurize(self, stream):
        sample = []
        pairs = self.reader.read_sentences(stream)
        for s1, s2 in pairs:
            pair = SentencePair(s1, s2)
            self.aligner.align(pair)
            sample.append(pair)
        return sample

    def convert_to_table(self, sample):
        table = []
        for s in sample:
            table.append([0] * self._feat_i)
            for feat, sc in s.features.iteritems():
                if not self._feat_order:
                    self._feat_order[feat] = 0
                    self._feat_i = 1
                    table[-1] = [sc]
                elif not feat in self._feat_order:
                    self._feat_order[feat] = self._feat_i
                    self._feat_i += 1
                    table[-1].append(sc)
                else:
                    table[-1][self._feat_order[feat]] = sc
        return table

    def predict_regression(self, feats, true_th=0.5):
        scores = []
        for sample in feats:
            ans = sum(self.model[i] * x for i, x in enumerate(sample))
            if self.conf.getboolean('regression', 'binary_labels'):
                f = 0 if ans < true_th else 1
            scores.append(f)
        return scores

    def print_results(self):
        try:
            gold_fn = self.conf.get('regression', 'gold')
            with open(gold_fn) as f:
                gold = self.read_labels(f)
        except:
            gold = None
            return
        stat = defaultdict(int)
        for i, g in enumerate(self.predicted):
            if g == 1:
                if gold[i] == 1:
                    stat['tp'] += 1
                else:
                    stat['fp'] += 1
            else:
                if gold[i] == 1:
                    stat['fn'] += 1
                else:
                    stat['tn'] += 1
        prec = stat['tp'] / float(stat['tp'] + stat['fp'])
        rec = stat['tp'] / float(stat['tp'] + stat['fn'])
        F = 2 * prec * rec / (prec + rec)
        print('prec: {0}\nrec: {1}\nF1: {2}'.format(prec, rec, F))
