from argparse import ArgumentParser
from sys import stdin
import logging
from ConfigParser import ConfigParser

from read_and_enrich import ReadAndEnrich
from align_and_penalize import AlignAndPenalize
from sentence import SentencePair
from resources import Resources
from regression import Regression


def parse_args():
    p = ArgumentParser()
    p.add_argument(
        '-c', '--conf', help='config file', default='config', type=str)
    return p.parse_args()


def read_config():
    args = parse_args()
    conf = ConfigParser()
    conf.read(args.conf)
    return conf


def main():
    conf = read_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    Resources.set_config(conf)
    Resources.ensure_nltk_packages()
    reader = ReadAndEnrich(conf)
    aligner = AlignAndPenalize(conf)
    if conf.get('final_score', 'mode') == 'regression':
        r = Regression(conf, reader, aligner)
        r.regression()
        r.print_results()
    else:
        pairs = reader.read_sentences(stdin)
        for i, (s1, s2) in enumerate(pairs):
            if i % 1000 == 0:
                logging.info('{0} pairs'.format(i))
            pair = SentencePair(s1, s2)
            print(aligner.align(pair))

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('main()', 'stats.cprofile')
    main()
