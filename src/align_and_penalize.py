from collections import defaultdict
import logging
import math
import os
import re
from sys import stderr, stdin

def log(s):
    stderr.write(s)
    stderr.flush()

log('loading wordnet...')
from nltk.corpus import wordnet
log('done\n')

from nltk.tag.hunpos import HunposTagger
from nltk.tokenize import word_tokenize

__EN_FREQ_PATH__ = '/mnt/store/home/hlt/Language/English/Freq/freqs.en'


class AlignAndPenalize(object):
    num_re = re.compile(r'^[0-9.,]+$', re.UNICODE)
    pronouns = {
        'i': 'me', 'me': 'i',
        'he': 'him', 'him': 'he',
        'she': 'her', 'her': 'she',
        'we': 'us', 'us': 'we',
        'they': 'them', 'them': 'they'}

    def __init__(self, sen1, sen2, tags1, tags2, map_tags, wrapper):
        self.wrapper = wrapper
        self.sen1 = []
        self.sen2 = []
        self.map_tags = map_tags
        for i, tok1 in enumerate(sen1):
            self.sen1.append({})
            self.sen1[-1]['token'] = tok1
            self.map_tags(tags1[i], self.sen1[-1])
        for i, tok2 in enumerate(sen2):
            self.sen2.append({})
            self.sen2[-1]['token'] = tok2
            self.map_tags(tags2[i], self.sen2[-1])

    @staticmethod
    def sts_map_tags(pos_tag, token_d):
        token_d['pos'] = pos_tag

    @staticmethod
    def twitter_map_tags(tags, token_d):
        sp = tags.split('/')[1:]
        token_d['ner'] = sp[0]
        token_d['pos'] = sp[1]
        token_d['chunk'] = sp[2]

    def get_senses(self):
        for t in self.sen1 + self.sen2:
            if t['token'] in self.wrapper.sense_cache:
                t['senses'] = self.wrapper.sense_cache[t['token']]
                continue
            senses = set([t['token']])
            wn_sen = wordnet.synsets(t['token'])
            if len(wn_sen) >= 10:
                n = float(len(wn_sen))
                for s in wn_sen:
                    w = s.name.split('.')[0]
                    s_sen = wordnet.synsets(w)
                    if len(s_sen) / n <= 1.0 / 3:
                        senses.add(w)
            self.wrapper.sense_cache[t['token']] = senses
            t['senses'] = senses

    def get_most_similar_tokens(self):
        for x_i, x in enumerate(self.sen1):
            score, most_sim = max((
                (self.sim_xy(x['token'], y['token'], x_i, y_i), y['token'])
                for y_i, y in enumerate(self.sen2)))
            x['most_sim_word'] = most_sim
            x['most_sim_score'] = score
        for y_i, y in enumerate(self.sen2):
            score, most_sim = max((
                (self.sim_xy(y['token'], x['token'], x_i, y_i), x['token'])
                for x_i, x in enumerate(self.sen1)))
            y['most_sim_word'] = most_sim
            y['most_sim_score'] = score

    def sim_xy(self, x, y, x_pos, y_pos):
        max1 = 0.0
        for sx in self.wrapper.sense_cache[x]:
            sim = self.similarity_wrapper(sx, y, x_pos, y_pos)
            if sim > max1:
                max1 = sim
        max2 = 0.0
        for sy in self.wrapper.sense_cache[y]:
            sim = self.similarity_wrapper(x, sy, x_pos, y_pos)
            if sim > max2:
                max2 = sim
        return max(max1, max2)

    def baseline_similarity(self, x, y, x_i, y_i):
        return bigram_dist_jaccard(x, y)

    def similarity_wrapper(self, x, y, x_i, y_i):
        if self.is_num_equivalent(x, y):
            return 1
        if self.is_pronoun_equivalent(x, y):
            return 1
        if self.is_acronym(x, y, x_i, y_i):
            return 1
        if self.is_headof(x, y, x_i, y_i):
            return 1
        if self.is_consecutive_match(x, y, x_i, y_i):
            return 1
        if self.is_oov(x) or self.is_oov(y):
            return self.bigram_sim(x, y)
        return self.lsa_sim(x, y, x_i, y_i)

    def lsa_sim(self, x, y, x_i, y_i):
        #TODO
        return self.bigram_sim(x, y)

    def is_num_equivalent(self, x, y):
        num_x = self.numerical(x)
        num_y = self.numerical(y)
        if num_x and num_y:
            return num_x == num_y
        return False

    def is_pronoun_equivalent(self, x, y):
        x_ = x.lower()
        y_ = y.lower()
        return (x_ in AlignAndPenalize.pronouns and
                y_ == AlignAndPenalize.pronouns[x_])

    def is_acronym(self, x, y, x_i, y_i):
        #TODO
        return False

    def is_headof(self, x, y, x_i, y_i):
        #TODO
        return False

    def is_consecutive_match(self, x, y, x_i, y_i):
        if x_i != len(self.sen1) - 1:
            two_word = x + '-' + self.sen1[x_i + 1]['token']
            if two_word == y:
                return True
        elif x_i != 0:
            two_word = self.sen1[x_i - 1]['token'] + '-' + x
            if two_word == y:
                return True
        if y_i != len(self.sen2) - 1:
            two_word = y + '-' + self.sen2[y_i + 1]['token']
            if two_word == x:
                return True
        elif y_i != 0:
            two_word = self.sen2[y_i - 1]['token'] + '-' + y
            if two_word == x:
                return True
        return False

    def is_oov(self, token):
        #TODO
        return False

    def bigram_sim(self, x, y):
        bigrams1 = set(get_ngrams(x, 2).iterkeys())
        bigrams2 = set(get_ngrams(y, 2).iterkeys())
        if not bigrams1 and not bigrams2:
            return 0
        return float(len(bigrams1 & bigrams2)) / (len(bigrams1) +
                                                  len(bigrams2))

    def numerical(self, token):
        m = AlignAndPenalize.num_re.match(token)
        if not m:
            return False
        return token.replace(',', '.').rstrip('0')

    def sentence_similarity(self):
        s1 = s2 = 0.0
        for tok1 in self.sen1:
            s1 += tok1['most_sim_score']
        for tok2 in self.sen2:
            s2 += tok2['most_sim_score']
        self.T = float(s1) / (2 * len(self.sen1)) + float(s2) / (2 * len(
                                                                 self.sen2))
        self.penalty()
        return self.T - self.P

    def weight_freq(self, token):
        if token in self.wrapper.global_freqs:
            return self.wrapper.global_freqs[token]
        return 0

    def weight_pos(self, token):
        #TODO
        if 'pos' in token:
            if token['pos'] in ['NN', 'NNP', 'VBZ']:
                return 1
        return 0

    def is_antonym(self, w1, w2):
        #TODO
        return False

    def penalty(self):
        A1 = [t for t in self.sen1 if t['most_sim_score'] < 0.05]
        A2 = [t for t in self.sen2 if t['most_sim_score'] < 0.05]
        B1 = set([(t['token'], t['most_sim_word'], t['most_sim_score'])
                 for t in self.sen1 if self.is_antonym(t['token'],
                                                       t['most_sim_word'])])
        B2 = set([(t['token'], t['most_sim_word'], t['most_sim_score'])
                 for t in self.sen2 if self.is_antonym(t['token'],
                                                       t['most_sim_word'])])
        P1A = 0.0
        for t in A1:
            score = t['most_sim_score']
            pos = t['pos']
            token = t['token']
            P1A += score + self.weight_freq(token) * self.weight_pos(pos)
        P1A /= float(2 * len(self.sen1))
        P2A = 0.0
        for t in A2:
            score = t['most_sim_score']
            pos = t['pos']
            token = t['token']
            P2A += score + self.weight_freq(token) * self.weight_pos(pos)
        P2A /= float(2 * len(self.sen2))
        P1B = 0.0
        for t, gt, score in B1:
            P1B += score + 0.5
        P2B = 0.0
        for t, gt, score in B2:
            P2B += score + 0.5
        self.P = P1A + P2A + P1B + P2B


def bigram_dist_jaccard(tok1, tok2):
    bigrams1 = set(get_ngrams(tok1, 2).iterkeys())
    bigrams2 = set(get_ngrams(tok2, 2).iterkeys())
    return jaccard(bigrams1, bigrams2)


def get_ngrams(text, N):
    ngrams = defaultdict(int)
    for i in xrange(len(text) - N + 1):
        ngram = text[i:i + N]
        ngrams[ngram] += 1
    return ngrams


def jaccard(s1, s2):
    try:
        return float(len(s1 & s2)) / len(s1 | s2)
    except ZeroDivisionError:
        return 0.0


class STSWrapper():
    def __init__(self):
        logging.info('reading global frequencies...')
        self.read_freqs()
        self.sense_cache = {}
        hunmorph_dir = os.environ['HUNMORPH_DIR']
        self.hunpos_tagger = HunposTagger(os.path.join(hunmorph_dir,
                                                       'en_wsj.model'))

    def parse_twitter_line(self, fd):
        sen1 = fd[2].split(' ')
        sen2 = fd[3].split(' ')
        tags1 = fd[5].split(' ')
        tags2 = fd[6].split(' ')
        return sen1, sen2, tags1, tags2

    def parse_sts_line(self, fields):
        sen1_toks, sen2_toks = map(word_tokenize, fields)
        sen1_pos, sen2_pos = map(self.hunpos_tagger.tag,
                                 (sen1_toks, sen2_toks))
        return sen1_toks, sen2_toks, sen1_pos, sen2_pos

    def read_freqs(self, ifn=__EN_FREQ_PATH__):
        self.global_freqs = {}
        with open(ifn) as f:
            for l in f:
                fd = l.decode('utf8').strip().split(' ')
                word = fd[0]
                freq = -math.log(int(fd[1]))
                self.global_freqs[word] = freq

    def process_line(self, line):
        fields = line.decode('utf8').strip().split('\t')
        if len(fields) == 7:
            parser = self.parse_twitter_line
            map_tags = AlignAndPenalize.twitter_map_tags
        elif len(fields) == 2:
            parser = self.parse_sts_line
            map_tags = AlignAndPenalize.sts_map_tags
        else:
            raise Exception('unknown input format: {0}'.format(fields))
        sen1, sen2, tags1, tags2 = parser(fields)
        aligner = AlignAndPenalize(sen1, sen2, tags1, tags2, map_tags,
                                   wrapper=self)
        aligner.get_senses()
        aligner.get_most_similar_tokens()
        print aligner.sentence_similarity()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    wrapper = STSWrapper()
    map(wrapper.process_line, stdin)

if __name__ == '__main__':
    main()
