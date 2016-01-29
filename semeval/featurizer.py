from sentence import Sentence, SentencePair
import cPickle
from read_and_enrich import ReadAndEnrich
from align_and_penalize import AlignAndPenalize
from ConfigParser import ConfigParser
from argparse import ArgumentParser
from resources import Resources
from numpy import array
import logging

def parse_args():
    p = ArgumentParser()
    p.add_argument(
        '-c', '--conf', help='config file', default='config', type=str)
    p.add_argument(
                '-i','--inputs', help='input list, separated by ,', type=str)
    p.add_argument(
        '-o' ,'--outputs', help='output list, separated by ,', type=str)
    return p.parse_args()


def read_config(args):
    conf = ConfigParser()
    conf.read(args.conf)
    return conf

class Featurizer(object):

    def __init__(self, conf):
        self.reader = ReadAndEnrich(conf)
        self.aligner = AlignAndPenalize(conf)
        self.conf = conf
        self._feat_order = {}
        self._feat_i = 0

    def featurize(self, stream):
        sample = []
        self.reader.pairs = []
        pairs = self.reader.read_sentences(stream)
        for s1, s2 in pairs:
            # fallback to no stopword removal
            if len(s1.tokens) == 0 or len(s2.tokens) == 0:
                s1_orig = Sentence(s1.sentence, s1.orig_tokens)
                s2_orig = Sentence(s2.sentence, s2.orig_tokens)
                pair = SentencePair(s1_orig, s2_orig)
            else:    
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
                elif feat not in self._feat_order:
                    self._feat_order[feat] = self._feat_i
                    self._feat_i += 1
                    table[-1].append(sc)
                else:
                    table[-1][self._feat_order[feat]] = sc
        return array(table)

    def dump_data(self, data, fn):
        fh = open(fn, 'w')
        d = {'data': data, 'config': self.conf, 'feats': self._feat_order}
        cPickle.dump(d, fh)

    def preproc_data(self, fn, output_fn):
        fh = open(fn)
        sample = self.featurize(fh)
        table = self.convert_to_table(sample)
        self.dump_data(table, output_fn)


def main():
    args = parse_args()
    conf = read_config(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    Resources.set_config(conf)
    Resources.ensure_nltk_packages()
    
    a = Featurizer(conf)
    inputs = args.inputs.split(',')
    outputs = args.outputs.split(',')
    for i, f in enumerate(inputs):
        of = outputs[i]
        a.preproc_data(f, of)
    exit()

if __name__ == "__main__":
    main()
