from argparse import ArgumentParser
from ConfigParser import ConfigParser
from read_and_enrich import ReadAndEnrich
from align_and_penalize import AlignAndPenalize
from sentence import SentencePair
from resources import Resources


def parse_args():
    p = ArgumentParser()
    p.add_argument('-c', '--conf', help='config file', default='config', type=str)
    return p.parse_args()


def read_config():
    args = parse_args()
    conf = ConfigParser()
    conf.read(args.conf)
    return conf


def main():
    conf = read_config()
    Resources.set_config(conf)
    reader = ReadAndEnrich(conf)
    pairs = reader.read_sentences()
    aligner = AlignAndPenalize(conf)
    for s1, s2 in pairs:
        pair = SentencePair(s1, s2)
        print(aligner.align(pair))

if __name__ == '__main__':
    main()
