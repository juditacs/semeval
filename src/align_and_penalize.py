from collections import defaultdict
from argparse import ArgumentParser
import HTMLParser
import logging
import math
import os
import re
import string
from sys import stderr, stdin

from utils import twitter_candidates
from gensim.models import Word2Vec

def log(s):
    stderr.write(s)
    stderr.flush()

import nltk

from nltk.corpus import wordnet

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tag.hunpos import HunposTagger

from hunspell_wrapper import HunspellWrapper
assert HunspellWrapper  # silence pyflakes

__EN_FREQ_PATH__ = '/mnt/store/home/hlt/Language/English/Freq/umbc_webbase.unigram_freq'  # nopep8
feats = []
global_freqs = {}


def dice(s1, s2):
    try:
        return (2 * float(len(s1 & s2))) / (len(s1) + len(s2))
    except ZeroDivisionError:
        return 0.0


def jaccard(s1, s2):
    try:
        return float(len(s1 & s2)) / len(s1 | s2)
    except ZeroDivisionError:
        return 0.0


global_flags = {
    'filter_stopwords': True,
    'penalize_antonyms': False,
    'penalize_questions': False,
    'penalize_named_entities': False,
    'wordnet_boost': True,
    'twitter_norm': False,
    'ngrams': 4,
    'ngram_padding': False,
    'ngram_sim': dice,
    'log_oov_stat': False,
    'verb_tense_penalty': False,
}


class AlignAndPenalize(object):
    num_re = re.compile(r'^([0-9][0-9.,]*)([mMkK]?)$', re.UNICODE)
    preferred_pos = ('VB', 'VBD', 'VBG', 'VBN', 'VBP',  # verbs
                     'NN', 'NNS', 'NNP', 'NNPS',   # nouns
                     'PRP', 'PRP$', 'CD')  # pronouns, numbers
    pronouns = {
        'me': 'i', 'my': 'i',
        'your': 'you',
        'him': 'he', 'his': 'he',
        'her': 'she',
        'us': 'we', 'our': 'we',
        'them': 'they', 'their': 'they',
    }

    question_starters = set([
        'is', 'does', 'do', 'what', 'where', 'how', 'why',
    ])

    written_numbers = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

    def __init__(self, sen1, sen2, tags1, tags2, map_tags, wrapper,
                 sim_function, wn_cache, hunspell_wrapper):
        #logging.debug('AlignAndPenalize init:')
        #logging.debug('sen1: {0}'.format(sen1))
        #logging.debug('sen2: {0}'.format(sen2))
        self.sts_wrapper = wrapper
        if isinstance(sim_function, str):
            self.sim_function = getattr(self, sim_function)
        else:
            self.sim_function = sim_function
        self.sen1 = []
        self.sen2 = []
        self.map_tags = map_tags
        self.wn_cache = wn_cache
        self.hunspell_wrapper = hunspell_wrapper
        for i, tok1 in enumerate(sen1):
            self.sen1.append({})
            self.sen1[-1]['token'] = tok1
            self.map_tags(tags1[i], self.sen1[-1])
        for i, tok2 in enumerate(sen2):
            self.sen2.append({})
            self.sen2[-1]['token'] = tok2
            self.map_tags(tags2[i], self.sen2[-1])

        #logging.info('sen1 unfiltered: {}'.format(self.sen1))
        #logging.info('sen2 unfiltered: {}'.format(self.sen2))
        self.sen1 = self.sts_wrapper.filter_sen(self.sen1)
        self.sen2 = self.sts_wrapper.filter_sen(self.sen2)
        #logging.info('sen1: {}'.format(self.sen1))
        #logging.info('sen2: {}'.format(self.sen2))

        self.compound_pairs = AlignAndPenalize.get_compound_pairs(self.sen1,
                                                                  self.sen2)

        self.acronym_pairs, self.head_pairs = (
            AlignAndPenalize.get_acronym_pairs(self.sen1, self.sen2))

        #logging.debug('compound pairs: {0}'.format(self.compound_pairs))

    @staticmethod
    def sts_map_tags((pos, ner), token_d):
        token_d['pos'] = pos
        token_d['ner'] = ner

    @staticmethod
    def twitter_map_tags(tags, token_d):
        sp = tags.split('/')[1:]
        token_d['ner'] = sp[0]
        token_d['pos'] = sp[1]
        token_d['chunk'] = sp[2]

    @staticmethod
    def get_acronym_pairs(sen1, sen2):
        acronym_pairs, head_pairs = set(), set()
        sen1_toks = [word['token'] for word in sen1]
        sen2_toks = [word['token'] for word in sen2]
        for src_sen, tgt_sen in ((sen1_toks, sen2_toks),
                                 (sen2_toks, sen1_toks)):
            for pair, is_head in AlignAndPenalize._get_acronym_pairs(
                    src_sen, tgt_sen):
                acronym_pairs.add(pair)
                if is_head:
                    head_pairs.add(pair)
        return acronym_pairs, head_pairs

    @staticmethod
    def _get_acronym_pairs(sen1, sen2):
        candidates = {}
        for i in range(len(sen2) - 1):
            for j in range(2, 5):
                if i + j > len(sen2):
                    continue
                words = sen2[i:i + j]
                abbr = "".join(w[0] for w in words)
                candidates[abbr] = words

        for word1 in sen1:
            if word1 in candidates:
                words2 = candidates[word1]
                for word2 in words2[:-1]:
                    yield (word2, word1), False
                    yield (word1, word2), False
                yield (words2[-1], word1), True
                yield (words2[-1], word1), True

    @staticmethod
    def get_compound_pairs(sen1, sen2):
        compound_pairs = set()
        sen1_toks = [word['token'] for word in sen1]
        sen2_toks = [word['token'] for word in sen2]
        for src_sen, tgt_sen in ((sen1_toks, sen2_toks),
                                 (sen2_toks, sen1_toks)):
            for pair in AlignAndPenalize._get_compound_pairs(src_sen, tgt_sen):
                compound_pairs.add(pair)
        return compound_pairs

    @staticmethod
    def _get_compound_pairs(sen1, sen2):
        #logging.debug('got these: {0}, {1}'.format(sen1, sen2))
        sen2_set = set(sen2)
        for i, tok in enumerate(sen1):
            if i == len(sen1) - 1:
                continue
            candidates = [pattern.format(tok, sen1[i + 1])
                          for pattern in (u"{0}{1}", u"{0}-{1}")]
            #logging.debug('tok: {0}, cands: {1}'.format(tok, candidates))
            for cand in candidates:
                if cand in sen2_set:
                    tgt_tok = (sen2.index(cand), cand)
                    for src_tok in (i, tok), (i + 1, sen1[i + 1]):
                        yield (src_tok, tgt_tok)
                        yield (tgt_tok, src_tok)

    def get_senses(self):
        for t in self.sen1 + self.sen2:
            t['senses'] = self.wn_cache.get_senses(t['token'])

    def get_most_similar_tokens(self):
        for x_i, x in enumerate(self.sen1):
            score, most_sim = max((
                (self.sim_xy(x['token'], y['token'], x_i, y_i), y['token'])
                for y_i, y in enumerate(self.sen2)))
            x['most_sim_word'] = most_sim
            x['most_sim_score'] = score
            logging.info(u'{0}. {1} -> {2} ({3})'.format(
                x_i, x['token'], most_sim, score))
        for y_i, y in enumerate(self.sen2):
            score, most_sim = max((
                (self.sim_xy(y['token'], x['token'], x_i, y_i), x['token'])
                for x_i, x in enumerate(self.sen1)))
            y['most_sim_word'] = most_sim
            y['most_sim_score'] = score
            logging.info(u'{0}. {1} -> {2} ({3})'.format(
                y_i, y['token'], most_sim, score))

    def sim_xy(self, x, y, x_pos, y_pos):
        max1 = 0.0
        best_pair_1 = None
        #logging.info(u'got this: {0}, {1}'.format(x, y))
        for sx in self.wn_cache.get_senses(x):
            sim = self.similarity_wrapper(sx, y, x_pos, y_pos)
            if sim > max1:
                max1 = sim
                best_pair_1 = (sx, y)
        max2 = 0.0
        best_pair_2 = None
        for sy in self.wn_cache.get_senses(y):
            sim = self.similarity_wrapper(x, sy, x_pos, y_pos)
            if sim > max2:
                max2 = sim
                best_pair_2 = (sy, x)

        if max1 > max2:
            sim, best_pair = max1, best_pair_1
        else:
            sim, best_pair = max2, best_pair_2
        logging.info('best pair: {0} ({1})'.format(sim, best_pair))
        return sim

    def baseline_similarity(self, x, y, x_i, y_i):
        n = global_flags['ngram']
        return ngram_dist_jaccard(x, y, n)

    def similarity_wrapper(self, x, y, x_i, y_i):
        if x == y:
            return 1
        if self.is_num_equivalent(x, y):
            logging.info(u'equivalent numbers: {0}, {1}'.format(x, y))
            return 1
        if self.is_pronoun_equivalent(x, y):
            logging.info(u'equivalent pronouns: {0}, {1}'.format(x, y))
            return 1
        if self.is_acronym(x, y, x_i, y_i):
            logging.info(u'acronym match: {0}, {1}'.format(x, y))
            return 1
        if self.is_headof(x, y, x_i, y_i):
            logging.info(u'head_of match: {0}, {1}'.format(x, y))
            return 1
        if self.is_consecutive_match(x, y, x_i, y_i):
            logging.info(u'consecutive match: {0}, {1}'.format(x, y))
            return 1

        sim = self.sim_function(x, y, x_i, y_i)

        if sim is None:
            sim = self.hunspell_sim(x, y, x_i, y_i)
        if sim is None:
            return AlignAndPenalize.ngram_sim(x, y, x_i, y_i)

        return sim

    def hunspell_sim(self, x, y, x_i, y_i):
        if self.hunspell_wrapper is None:
            return None
        for suggestion in self.hunspell_wrapper.get_suggestions(x):
            sim = self.sim_function(suggestion, y, x_i, y_i)
            if sim is not None:
                #logging.info(
                    #'hunspell correcting {0} to {1}'.format(x, suggestion))
                return sim
        for suggestion in self.hunspell_wrapper.get_suggestions(y):
            sim = self.sim_function(x, suggestion, x_i, y_i)
            if sim is not None:
                #logging.info(
                    #'hunspell correcting {0} to {1}'.format(y, suggestion))
                return sim

    def lsa_sim(self, x, y, x_i, y_i):
        #TODO
        if self.is_oov(x) or self.is_oov(y):
            return None
        return AlignAndPenalize.ngram_sim(x, y, x_i, y_i)

    def is_num_equivalent(self, x, y):
        num_x = self.numerical(x)
        num_y = self.numerical(y)
        #logging.info("{0} converted to {1}".format(x, num_x))
        #logging.info("{0} converted to {1}".format(y, num_y))
        if num_x and num_y:
            return num_x == num_y
        return False

    def is_pronoun_equivalent(self, x, y):
        x_ = AlignAndPenalize.pronouns.get(x.lower(), x.lower())
        y_ = AlignAndPenalize.pronouns.get(y.lower(), y.lower())
        return x_ == y_

    def is_acronym(self, x, y, x_i, y_i):
        return (x, y) in self.acronym_pairs

    def is_headof(self, x, y, x_i, y_i):
        return (x, y) in self.head_pairs

    def is_consecutive_match(self, x, y, x_i, y_i):
        """We don't distinguish between sen1->sen2 or sen2->sen1, technically
        this could cause false positives, hence the position indices, this way
        it's _virtually_ impossible"""
        return ((x_i, x), (y_i, y)) in self.compound_pairs

    def is_consecutive_match_old(self, x, y, x_i, y_i):
        if x_i != len(self.sen1) - 1:
            two_word = x + '-' + self.sen1[x_i + 1]['token']
            if two_word == y:
                return True
        if x_i != 0:
            two_word = self.sen1[x_i - 1]['token'] + '-' + x
            if two_word == y:
                return True
        if y_i != len(self.sen2) - 1:
            two_word = y + '-' + self.sen2[y_i + 1]['token']
            if two_word == x:
                return True
        if y_i != 0:
            two_word = self.sen2[y_i - 1]['token'] + '-' + y
            if two_word == x:
                return True
        return False

    def is_oov(self, token):
        #TODO
        return False

    @staticmethod
    def ngram_sim(x, y, x_i, y_i):
        n = global_flags['ngrams']
        ngrams1 = set(get_ngrams(x, n).iterkeys())
        ngrams2 = set(get_ngrams(y, n).iterkeys())
        if not ngrams1 and not ngrams2:
            return 0
        sim_metric = global_flags['ngram_sim']
        return sim_metric(ngrams1, ngrams2)

    def numerical(self, token):
        if token in AlignAndPenalize.written_numbers:
            return AlignAndPenalize.written_numbers[token]
        m = AlignAndPenalize.num_re.match(token)
        if not m:
            return False
        num = float(m.group(1).replace(',', ''))
        if m.group(2):
            c = m.group(2).lower()
            if c == 'k':
                num *= 1000
            else:
                num *= 1000000
        return num

    def sentence_similarity(self):
        s1 = s2 = 0.0
        for tok1 in self.sen1:
            s1 += tok1['most_sim_score']
        for tok2 in self.sen2:
            s2 += tok2['most_sim_score']
        self.T = float(s1) / (2 * len(self.sen1)) + float(s2) / (2 * len(
                                                                 self.sen2))
        if self.T > 1:
            raise Exception(
                'alignment score > 1: {0} {1}'.format(self.sen1, self.sen2))
        self.penalty()
        #out = [len(self.sen1), len(self.sen2)]
        out = []
        for k, v in sorted(self.pen_feats.items()):
            out.append(v)
        feats.append(out)
        logging.info('T={0}, P={1}'.format(self.T, self.P))
        #sim = self.T if self.T >= 0.55 else self.T - self.P
        #sim = self.T - self.P*(max(0, 0.55-self.T) / 0.55)
        sim = self.T - self.P
        sim = sim if sim > 0 else 0
        return sim

    def weight_freq(self, token):
        if token in global_freqs:
            return 1 / global_freqs[token]
        return 1 / math.log(2)

    def weight_pos(self, pos):
        multiplier = 0
        if pos.upper() in AlignAndPenalize.preferred_pos:
            #logging.info('preferred pos: {0}'.format(pos))
            return multiplier
        #logging.info('not preferred pos: {0}'.format(pos))
        return 0.5 * multiplier

    def is_antonym(self, w1, w2):
        if w1 in self.sts_wrapper.antonym_cache(w2):
            logging.info('Antonym found: {0} -- {1}'.format(w1, w2))
            return True
        if w2 in self.sts_wrapper.antonym_cache(w1):
            logging.info('Antonym found: {0} -- {1}'.format(w2, w1))
            return True
        return False

    def antonym_penalty(self):
        B1 = set([(t['token'], t['most_sim_word'], t['most_sim_score'])
                 for t in self.sen1 if self.is_antonym(t['token'],
                                                       t['most_sim_word'])])
        B2 = set([(t['token'], t['most_sim_word'], t['most_sim_score'])
                 for t in self.sen2 if self.is_antonym(t['token'],
                                                       t['most_sim_word'])])
        P1B = 0.0
        for t, gt, score in B1:
            P1B += score + 0.5
        P1B /= float(2 * len(self.sen1))
        P2B = 0.0
        for t, gt, score in B2:
            P2B += score + 0.5
        P2B /= float(2 * len(self.sen2))
        return P1B, P2B

    def penalty(self):
        self.pen_feats = {}
        A1 = [t for t in self.sen1 if t['most_sim_score'] < 0.05]
        A2 = [t for t in self.sen2 if t['most_sim_score'] < 0.05]
        #logging.debug('A1: {0} words, A2: {1} words'.format(len(A1), len(A2)))
        if global_flags['penalize_antonyms']:
            P1B, P2B = self.antonym_penalty()
            self.pen_feats['P1B'] = P1B
            self.pen_feats['P2B'] = P2B
        else:
            P1B = P2B = 0
        P1A = 0.0
        for t in A1:
            score = t['most_sim_score']
            pos = t['pos']
            token = t['token']
            P1A += score + self.weight_freq(token) * self.weight_pos(pos)
            #logging.info('penalty for {0}: wf: {1}, wp: {2}'.format(
                #(token, pos), self.weight_freq(token), self.weight_pos(pos)))
        self.pen_feats['P1A'] = P1A
        P1A /= float(2 * len(self.sen1))
        P2A = 0.0
        for t in A2:
            score = t['most_sim_score']
            pos = t['pos']
            token = t['token']
            P2A += score + self.weight_freq(token) * self.weight_pos(pos)
            #logging.info('penalty for {0}: wf: {1}, wp: {2}'.format(
                #(token, pos), self.weight_freq(token), self.weight_pos(pos)))
        self.pen_feats['P2A'] = P2A
        P2A /= float(2 * len(self.sen2))
        if global_flags['penalize_named_entities']:
            PC = self.ne_penalty()
            self.pen_feats['PC'] = PC
        else:
            PC = 0
        if global_flags['penalize_questions']:
            PD = self.question_penalty()
            self.pen_feats['PD'] = PD
            PD /= (len(self.sen1) + len(self.sen2))
        else:
            PD = 0
        if global_flags['verb_tense_penalty']:
            PE = self.verb_tense_penalty()
            self.pen_feats['PE'] = PE
        else:
            PE = 0
        #logging.info('NE penalty: {0}'.format(PC))
        PC /= sum([len(self.sen1), len(self.sen2)])
        self.P = P1A + P2A + P1B + P2B + PC + PD + PE
        #logging.info('P1A: {0} P2A: {1} P1B: {2} P2B: {3}, PC: {4}'.format(
            #P1A, P2A, P1B, P2B, PC))
        if self.P < 0:
            raise Exception(
                'negative penalty: {0}\n'.format(self.P) +
                'sen1: {0}, sen2: {1}'.format(self.sen1, self.sen2))

    def verb_tense_penalty(self):
        past = set(['vbd', 'vbn'])
        is_past1 = False
        for tok in self.sen1:
            if tok['pos'].lower() in past:
                is_past1 = True
        is_past2 = False
        for tok in self.sen2:
            if tok['pos'].lower() in past:
                is_past2 = True
        if is_past1 == is_past2:
            return 0
        #logging.info('Different verb tense found')
        return 1

    def question_penalty(self):
        isq1 = (self.sen1[0]['token'].lower() in
                AlignAndPenalize.question_starters)
        isq2 = (self.sen2[0]['token'].lower() in
                AlignAndPenalize.question_starters)
        if isq1 == isq2:
            return 0
        return 1

    def find_ne_in_other(self, ne1, ne2):
        match = set()
        missing = set()
        for typ, nes in ne1.iteritems():
            if not typ in ne2:
                missing |= set(nes)
                continue
            for ne in nes:
                if ne in ne2[typ]:
                    match.add(ne)
                    continue
                words = ne.split(' ')
                for i in range(1, len(words) - 1):
                    p1 = ' '.join(words[:i])
                    p2 = ' '.join(words[i:])
                    if p1 in ne2[typ] or p2 in ne2[typ]:
                        match.add(ne)
                        continue
                missing.add(ne)
        return match, missing

    def ne_penalty(self):
        ne1, ne2 = self.collect_entities()
        self.pen_feats['match1'] = 0
        self.pen_feats['match2'] = 0
        self.pen_feats['missing1'] = 0
        self.pen_feats['missing2'] = 0
        self.pen_feats['ne1'] = sum(len(v) for v in ne1.values())
        self.pen_feats['ne2'] = sum(len(v) for v in ne2.values())
        if not ne1 and not ne2:
            return 0
        #logging.info('NE1: {0}, NE2: {1}'.format(ne1, ne2))
        match1, missing1 = self.find_ne_in_other(ne1, ne2)
        match2, missing2 = self.find_ne_in_other(ne2, ne1)
        self.pen_feats['match1'] = len(match1)
        self.pen_feats['match2'] = len(match2)
        self.pen_feats['missing1'] = len(missing1)
        self.pen_feats['missing2'] = len(missing2)
        if not match1 and not match2:
            return 1.0
        diff1 = float(len(match1 - match2)) / len(match1 | match2)
        diff2 = float(len(match2 - match1)) / len(match1 | match2)
        return 1 - max([diff1, diff2])

    def collect_entities(self):
        current_ne = []
        typ = ''
        ne1 = defaultdict(list)
        ne2 = defaultdict(list)
        for tok in self.sen1:
            if tok['ner'].startswith('b'):
                if current_ne:
                    ne1[typ].append(' '.join(current_ne))
                typ = tok['ner'].split('-')[1]
                current_ne = [tok['token']]
            elif not tok['ner'] == 'o':
                typ = tok['ner'].split('-')[1]
                current_ne.append(tok['token'])
        if current_ne:
            ne1[typ].append(' '.join(current_ne))
        current_ne = []
        for tok in self.sen2:
            if tok['ner'].startswith('b'):
                if current_ne:
                    ne2[typ].append(' '.join(current_ne))
                typ = tok['ner'].split('-')[1]
                current_ne = [tok['token']]
            elif not tok['ner'] == 'o':
                typ = tok['ner'].split('-')[1]
                current_ne.append(tok['token'])
        if current_ne:
            ne2[typ].append(' '.join(current_ne))
        return ne1, ne2


def ngram_dist_jaccard(tok1, tok2, n=2):
    ngrams1 = set(get_ngrams(tok1, n).iterkeys())
    ngrams2 = set(get_ngrams(tok2, n).iterkeys())
    return jaccard(ngrams1, ngrams2)


def get_ngrams(text, N):
    ngrams = defaultdict(int)
    if global_flags['ngram_padding']:
        padding = '@'
        #padding = '@'*(N-2)
        text = "{0}{1}{2}".format(padding, text, padding)
    for i in xrange(len(text) - N + 1):
        ngram = text[i:i + N]
        ngrams[ngram] += 1
    #logging.info('ngrams: {0} -> {1}'.format(text, ngrams))
    return ngrams


class LSAWrapper(object):

    def __init__(self, vector_fn='vectors_example.bin', word2vec=True):
        if word2vec:
            self.lsa_model = Word2Vec.load_word2vec_format(
                os.path.join(os.environ['LSA_DIR'], vector_fn), binary=True)
        else:
            self.lsa_model = Word2Vec.load(
                os.path.join(os.environ['LSA_DIR'], vector_fn))
        self.alpha = 0.25
        self.cache = {}
        self.wn_cache = WordnetCache()

    def wordnet_boost(self, word1, word2):
        sigsets1 = set(self.wn_cache.get_significant_synsets(word1))
        sigsets2 = set(self.wn_cache.get_significant_synsets(word2))
        if sigsets1 & sigsets2:
            # the two words appear together in the same synset
            return 0
        if self.is_direct_hypernym(sigsets1, sigsets2):
            #logging.info('Direct hypernym: {0} -- {1} -- score: 1'.format(
                #word1.encode('utf8'), word2.encode('utf8')))
            return 1
        if self.is_two_link_indirect_hypernym(sigsets1, sigsets2):
            return 2
        adj1 = set(filter(lambda x: x.pos() == 'a', sigsets1))
        adj2 = set(filter(lambda x: x.pos() == 'a', sigsets1))
        if adj1 and adj2:
            if self.is_direct_similar_to(adj1, adj2):
                #logging.info(
                    #'Direct similar to: {0} -- {1} -- score: 1'.format(
                        #word1.encode('utf8'), word2.encode('utf8')))
                return 1
            if self.is_two_link_indirect_similar_to(adj1, adj2):
                return 2
        if self.is_derivationally_related(sigsets1, sigsets2):
            #logging.info('Derivationally rel: {0} -- {1} -- score: 1'.format(
                #word1.encode('utf8'), word2.encode('utf8')))
            return 1
        """
        TODO
        word is the head of the gloss of the other or its direct hypernym or
        one of its direct hyponyms word appears frequently in the gloss of the
        other or its direct hypernym or one of its direct hyponyms see
        Collins, 1999
        """
        if self.in_hypopernym_glosses(word1, word2, sigsets1, sigsets2):
            return 2
        return None

    def in_hypopernym_glosses(self, word1, word2, synsets1, synsets2):
        if self.in_glosses(word1, synsets2):
            #logging.info(u'Word [{0}] in glosses of word [{1}]'.format(
                #word1, word2).encode('utf8'))
            return True
        if self.in_glosses(word2, synsets1):
            #logging.info(u'Word [{0}] in glosses of word [{1}]'.format(
                #word2, word1).encode('utf8'))
            return True

    def in_glosses(self, word, synsets):
        defs = defaultdict(int)
        for s in synsets:
            for w in s.definition():
                defs[w] += 1
            for h in s.hypernyms():
                for w in h.definition():
                    defs[w] += 1
            for h in s.hyponyms():
                for w in h.definition():
                    defs[w] += 1
        top5 = [i[0] for i in sorted(defs.iteritems(),
                                     key=lambda x: -x[1])[:5]]
        if word in top5:
            return True
        return False

    def is_derivationally_related(self, synsets1, synsets2):
        return False

        lemmas1 = set()
        der1 = set()
        for s1 in synsets1:
            lemmas1 |= set(s1.lemmas())
        lemmas2 = set()
        for s2 in synsets2:
            lemmas2 |= set(s2.lemmas())
        for l1 in lemmas1:
            der1 |= set(l1.derivationally_related_forms())
        if der1 & lemmas2:
            return True
        der2 = set()
        for l2 in lemmas2:
            der2 |= set(l2.derivationally_related_forms())
        if der2 & lemmas1:
            return True
        return False

    def wn_freq(self, synset):
        return sum(l.count() for l in synset.lemmas())

    def is_two_link_indirect_similar_to(self, adj1, adj2):
        sim1 = set()
        for s in adj1:
            sim1 |= s.two_link_similar_tos()
        if adj2 & sim1:
            return True
        sim2 = set()
        for s in adj2:
            sim2 |= s.two_link_similar_tos()
        if adj1 & sim2:
            return True
        return False

    def is_direct_similar_to(self, adj1, adj2):
        sim1 = set()
        for s in adj1:
            sim1 |= set(s.similar_tos())
        if sim1 & adj2:
            return True
        sim2 = set()
        for s in adj2:
            sim2 |= set(s.similar_tos())
        if sim2 & adj1:
            return True
        return False

    def is_two_link_indirect_hypernym(self, synsets1, synsets2):
        hyps1 = set()
        for s in synsets1:
            hyps1 |= s.two_link_hypernyms()
        if synsets2 & hyps1:
            return True
        hyps2 = set()
        for s in synsets2:
            hyps2 |= s.two_link_hypernyms()
        if synsets1 & hyps2:
            return True
        return False

    def is_direct_hypernym(self, synsets1, synsets2):
        hyps1 = set()
        for s1 in synsets1:
            hyps1 |= set(s1.hypernyms())
        if synsets2 & hyps1:
            return True
        hyps2 = set()
        for s2 in synsets2:
            hyps2 |= set(s2.hypernyms())
        if synsets1 & hyps2:
            return True
        return False

    def is_oov(self, word):
        if len(self.spell_candidates(word)) == 0:
            return True
        return False

    def spell_candidates(self, word):
        try:
            self.lsa_model[word]
            oov_stat['non_oov'].add(word)
            return [word]
        except KeyError:
            if global_flags['twitter_norm']:
                cand = twitter_candidates(word, self.lsa_model, oov_stat)
                if cand:
                    return cand
            oov_stat['oov'].add(word)
            return []

    def word_similarity(self, word1, word2, pos1, pos2):
        d = self.lookup_cache(word1, word2)
        if not d is None:
            return d
        cand1 = self.spell_candidates(word1)
        if len(cand1) == 0:
            return None
        cand2 = self.spell_candidates(word2)
        if len(cand2) == 0:
            return None
        max_sim = -1
        max_pair = (word1, word2)
        for c1 in cand1:
            for c2 in cand2:
                sim = self.lsa_model.similarity(c1, c2)
                if sim > max_sim:
                    max_sim = sim
                    #max_pair = (c1, c2)
        sim = max_sim
        if sim < 0.1:
            #logging.debug(u'LSA sim too low (less than 0.1), ' +
                          #'setting it to 0.0: {0} -- {1} -- {2}'.format(
                              #word1, word2, sim).encode('utf8'))
            return 0.0
        #logging.debug(u'LSA sim without wordnet: {0} -- {1} -- {2}'.format(
            #word1, word2, sim).encode('utf8'))
        if global_flags['wordnet_boost']:
            D = self.wordnet_boost(max_pair[0], max_pair[1])
            if D is not None:
                #logging.debug(
                    #u'LSA sim wordnet boost: {0} -- {1} -- {2}'.format(
                        #word1, word2, D).encode('utf8'))
                sim = sim + 0.5 * math.exp(-self.alpha * D)
            #logging.debug(u'LSA sim + wn boost: {0} -- {1} -- {2}'.format(
                #word1, word2, sim).encode('utf8'))
        d = sim if sim <= 1 else 1
        d = d if d >= 0 else 0
        self.store_cache(word1, word2, d)
        return d

    def lookup_cache(self, word1, word2):
        if not word1 in self.cache:
            return None
        if not word2 in self.cache[word1]:
            return None
        return self.cache[word1][word2]

    def store_cache(self, word1, word2, score):
        if not word1 in self.cache:
            self.cache[word1] = {}
        self.cache[word1][word2] = score


class SynsetWrapper(object):

    punct_re = re.compile(r'[\(\)]', re.UNICODE)
    nltk_sw = set(nltk_stopwords.words('english')) - set(
        AlignAndPenalize.pronouns.iterkeys())

    def __init__(self, synset):
        self.synset = synset
        self._lemmas = None
        self._freq = None
        self._hypernyms = None
        self._hyponyms = None
        self._two_link_hypernyms = None
        self._pos = None
        self._similar_tos = None
        self._two_link_similar_tos = None
        self._definition = None

    def __hash__(self):
        return hash(self.synset)

    def definition(self):
        if self._definition is None:
            def_ = self.synset.definition()
            def_ = SynsetWrapper.punct_re.sub(' ', def_)
            self._definition = set(
                [w.strip() for w in def_.split() if (
                    w.strip() and not w.strip() in SynsetWrapper.nltk_sw)])
        return self._definition

    def freq(self):
        if self._freq is None:
            self._freq = 0
            for lemma in self.lemmas:
                self._freq += lemma.count()
        return self._freq

    def lemmas(self):
        if self._lemmas is None:
            self._lemmas = []
            for lemma in self.synset.lemmas():
                self._lemmas.append(lemma)
        return self._lemmas

    def hyponyms(self):
        if self._hyponyms is None:
            self._hyponyms = set()
            for h in self.synset.hyponyms():
                self._hyponyms.add(SynsetWrapper(h))
        return self._hyponyms

    def hypernyms(self):
        if self._hypernyms is None:
            self._hypernyms = set()
            for h in self.synset.hypernyms():
                self._hypernyms.add(SynsetWrapper(h))
        return self._hypernyms

    def two_link_hypernyms(self):
        if self._two_link_hypernyms is None:
            self._two_link_hypernyms = set()
            for hyp in self.hypernyms():
                self._two_link_hypernyms |= hyp.hypernyms()
        return self._two_link_hypernyms

    def pos(self):
        if self._pos is None:
            self._pos = self.synset.pos()
        return self._pos

    def similar_tos(self):
        if self._similar_tos is None:
            self._similar_tos = set(
                SynsetWrapper(s) for s in self.synset.similar_tos())
        return self._similar_tos

    def two_link_similar_tos(self):
        if self._two_link_similar_tos is None:
            self._two_link_similar_tos = set()
            for s in self.similar_tos():
                self._two_link_similar_tos |= s.similar_tos()
        return self._two_link_similar_tos


class WordnetCache(object):

    def __init__(self):
        self.synsets = {}
        self.synset_to_wrapper = {}
        self.senses = {}

    def get_significant_synsets(self, word):
        if not word in self.synsets:
            candidates = wordnet.synsets(word)
            if len(candidates) == 0:
                self.synsets[word] = []
            else:
                sn = SynsetWrapper(candidates[0])
                self.synset_to_wrapper[candidates[0]] = sn
                self.synsets[word] = [sn]
                for c in candidates[1:]:
                    sw = SynsetWrapper(c)
                    self.synset_to_wrapper[c] = sw
                    if sw.freq >= 5:
                        self.synsets[word].append(sw)
                        continue
                    if sw.lemmas()[0].name() == word and len(sw.lemmas()) < 8:
                        self.synsets[word].append(sw)
        return self.synsets[word]

    def get_senses(self, word):
        if not word in self.senses:
            self.senses[word] = set([word])
            sn = wordnet.synsets(word)
            if len(sn) >= 100:
                th = len(sn) / 3.0
                for synset in sn:
                    for lemma in synset.lemmas():
                        lsn = wordnet.synsets(lemma.name())
                        if len(lsn) <= th:
                            self.senses[word].add(
                                lemma.name().replace('_', ' '))
            #logging.info('Synonyms for word [{0}]: {1}'.format(
                #word.encode('utf8'), self.senses[word]))
        return self.senses[word]


class STSWrapper(object):

    custom_stopwords = set([])
    #custom_stopwords = set(["'s"])
    punctuation = set(string.punctuation)
    punct_regex = re.compile("\W+")

    def __init__(self, sim_function='lsa_sim', wn_cache=None,
                 hunspell_wrapper=None):
        #logging.info('reading global frequencies...')
        self.sim_function = sim_function
        self.read_freqs()
        self.sense_cache = {}
        self.frequent_adverbs_cache = {}
        self.hunpos_tagger = STSWrapper.get_hunpos_tagger()
        self.html_parser = HTMLParser.HTMLParser()
        if global_flags['filter_stopwords']:
            self.stopwords = STSWrapper.get_stopwords()
        else:
            self.stopwords = set()
        self.hunspell_wrapper = hunspell_wrapper
        self._antonym_cache = {}
        if wn_cache:
            self.wn_cache = wn_cache
        else:
            self.wn_cache = WordnetCache()

    def antonym_cache(self, key):
        if not key in self._antonym_cache:
            self._antonym_cache[key] = set()
            for synset in wordnet.synsets(key):
                for lemma in synset.lemmas():
                    for antonym in lemma.antonyms():
                        self._antonym_cache[key].add(
                            antonym.name().split('.')[0])
        return self._antonym_cache[key]

    @staticmethod
    def get_hunpos_tagger():
        hunmorph_dir = os.environ['HUNMORPH_DIR']
        hunpos_binary = os.path.join(hunmorph_dir, 'hunpos-tag')
        hunpos_model = os.path.join(hunmorph_dir, 'en_wsj.model')
        return HunposTagger(hunpos_model, hunpos_binary)

    @staticmethod
    def get_stopwords():
        nltk_sw = set(nltk_stopwords.words('english')) - set(
            AlignAndPenalize.pronouns.iterkeys())
        return nltk_sw.union(STSWrapper.custom_stopwords)

    def parse_twitter_test_line(self, fd):
        sen1 = fd[2].split(' ')
        sen2 = fd[3].split(' ')
        tags1 = fd[4].split(' ')
        tags2 = fd[5].split(' ')
        return sen1, sen2, tags1, tags2

    def parse_twitter_line(self, fd):
        sen1 = fd[2].split(' ')
        sen2 = fd[3].split(' ')
        tags1 = fd[5].split(' ')
        tags2 = fd[6].split(' ')
        return sen1, sen2, tags1, tags2

    def tokenize(self, sen):
        toks = nltk.word_tokenize(self.html_parser.unescape(sen))
        new_toks = []
        for tok in toks:
            if tok in STSWrapper.punctuation:
                new_toks.append(tok)
            else:
                new_toks += STSWrapper.punct_regex.split(tok)

        return filter(lambda w: w not in ("", "s"), new_toks)

    def nva_filter(self, sen):
        for cat_filter in (("N", "V", "J", "C", "R"),):
        #for cat_filter in (("N", "V"),):
        #for cat_filter in (("N", "V"), ("N", "V", "J")):
            filtered = [word for word in sen if word['pos'][0] in cat_filter]
            if len(filtered) > 0:
                return filtered
        else:
            return sen

    def filter_sen(self, sen):
        #nva_filtered = self.nva_filter(sen)
        nva_filtered = sen
        filtered = [word for word in nva_filtered if (
            word['token'] not in STSWrapper.punctuation and
            word['token'].lower() not in self.stopwords and
            not self.is_frequent_adverb(word['token'], word['pos']))]
        if filtered:
            return filtered
        return nva_filtered

    def get_tags_from_ne(self, ne):
        tags = []
        for piece in ne:
            if isinstance(piece, tuple):
                tok, pos = piece
                tags.append((pos, 'o'))
            else:
                ne_type = piece.label()
                tags.append((piece[0][1], "b-{0}".format(ne_type)))
                tags += [(tok[1], "i-{0}".format(ne_type))
                         for tok in piece[1:]]

        return tags

    def parse_sts_line(self, fields):
        sen1_toks, sen2_toks = map(self.tokenize, fields)
        #logging.info("tokenized: {0}".format((sen1_toks, sen2_toks)))
        sen1_pos, sen2_pos = map(self.hunpos_tagger.tag,
                                 (sen1_toks, sen2_toks))
        #logging.info("pos-tagged: {0}".format((sen1_pos, sen2_pos)))
        sen1_ne, sen2_ne = map(nltk.ne_chunk, (sen1_pos, sen2_pos))
        sen1_toks, sen2_toks = map(lambda l: [w.lower() for w in l],
                                   (sen1_toks, sen2_toks))
        tags1, tags2 = map(self.get_tags_from_ne, (sen1_ne, sen2_ne))
        return sen1_toks, sen2_toks, tags1, tags2

    def read_freqs(self, ifn=__EN_FREQ_PATH__):
        global global_freqs
        if len(global_freqs) > 0:
            #logging.info('Skipping global freq reading')
            return
        with open(ifn) as f:
            for l in f:
                try:
                    fd = l.decode('utf8').strip().split(' ')
                    word = fd[1]
                except:
                    logging.warning(
                        "error reading line in freq data: {0}".format(repr(l)))
                    continue
                logfreq = math.log(int(fd[0]) + 2)
                #we add 2 so we can calculate inverse logfreq for OOVs
                global_freqs[word] = logfreq

    def is_frequent_adverb(self, word, pos):
        answer = self.frequent_adverbs_cache.setdefault(
            (pos is not None and pos[:2] == 'RB' and
             global_freqs.get(word, 2) > 500000))
        #if answer:
            #logging.warning("discarding frequent adverb: {0}".format(word))
        return answer

    def process_line(self, line):
        fields = line.decode('latin1').strip().split('\t')
        if args.shell:
            parser = parse_interactive_input
            map_tags = dummy_map
        elif len(fields) == 6:
            parser = self.parse_twitter_test_line
            map_tags = AlignAndPenalize.twitter_map_tags
        elif len(fields) == 7:
            parser = self.parse_twitter_line
            map_tags = AlignAndPenalize.twitter_map_tags
        elif len(fields) == 2:
            parser = self.parse_sts_line
            map_tags = AlignAndPenalize.sts_map_tags
        else:
            raise Exception('unknown input format: {0}'.format(fields))
        sen1, sen2, tags1, tags2 = parser(fields)
        aligner = AlignAndPenalize(sen1, sen2, tags1, tags2, map_tags,
                                   wrapper=self,
                                   sim_function=self.sim_function,
                                   wn_cache=self.wn_cache,
                                   hunspell_wrapper=self.hunspell_wrapper)
        aligner.get_senses()
        aligner.get_most_similar_tokens()
        print aligner.sentence_similarity()
        if args.full_test:
            return aligner.T


def dummy_map(tags, token_d):
    token_d['ner'] = ''
    token_d['chunk'] = ''
    token_d['pos'] = ''


def parse_interactive_input(fields):
    sen1 = fields[0].split(' ')
    sen2 = fields[1].split(' ')
    return sen1, sen2, [{}] * len(sen1), [{}] * len(sen2)


class HybridSimWrapper():

    def __init__(self, lsa_wrapper, machine_sim):
        self.lsa_wrapper = lsa_wrapper
        self.machine_sim = machine_sim

    def lsa_first_sim(self, x, y, x_i, y_i):
        lsa_sim = self.lsa_wrapper.word_similarity(x, y, x_i, y_i)
        if lsa_sim is not None:
            return lsa_sim
        return self.machine_sim.word_similarity(x, y, x_i, y_i)

    def machine_first_sim(self, x, y, x_i, y_i):
        machine_sim = self.machine_sim.word_similarity(x, y, x_i, y_i)
        if machine_sim is not None:
            return machine_sim
        return self.lsa_wrapper.word_similarity(x, y, x_i, y_i)

    def max_sim(self, x, y, x_i, y_i):
        machine_sim = self.machine_sim.word_similarity(x, y, x_i, y_i)
        lsa_sim = self.lsa_wrapper.word_similarity(x, y, x_i, y_i)
        ngram_sim = AlignAndPenalize.ngram_sim(x, y, x_i, y_i)
        max_sim = max((machine_sim, lsa_sim, ngram_sim))
        #logging.info("max sim: {0} vs. {1}: {2}".format(x, y, max_sim))
        return max_sim

    def average_sim(self, x, y, x_i, y_i):
        machine_sim = self.machine_sim.word_similarity(x, y, x_i, y_i)
        lsa_sim = self.lsa_wrapper.word_similarity(x, y, x_i, y_i)
        if machine_sim is None:
            if lsa_sim is None:
                sim = None
            else:
                sim = lsa_sim
        elif lsa_sim is None:
            sim = machine_sim
        else:
            sim = (machine_sim + lsa_sim) / 2
            #sim = min(machine_sim, lsa_sim)

        return sim


def parse_args():
    p = ArgumentParser()
    p.add_argument('--sim-type', help='similarity type', type=str)
    p.add_argument('--shell', help='interactive shell',
                   action='store_true', default=False)
    p.add_argument('--vectors', help='vectors file or prefix', type=str,
                   default='gensim_vec')
    p.add_argument('--word2vec', help='word2vec or gensim vectors',
                   action='store_true', default=False)
    p.add_argument('--batch', help='use batch processing', action='store_true',
                   default=False)
    p.add_argument('--lower', action='store_true', default=False,
                   help='lower all input')
    p.add_argument('--synonyms', type=str)
    p.add_argument('--full-test',
                   help='try all similarity methods and output the results in a feture file',  # nopep8
                   action='store_true', default=False)
    p.add_argument('--features', type=str, default='features',
                   help='output features to file')
    return p.parse_args()


def synonym_sim(synonyms, word1, word2):
    if word1 in synonyms and word2 in synonyms[word1]:
        #logging.info('Synonyms: {0} - {1}'.format(word1, word2))
        return 1.0
    return 0.0


def read_synonyms(fn):
    synonyms = defaultdict(set)
    import gzip
    with gzip.open(fn) as f:
        for l in f:
            fs = l.decode('utf8').strip().split('\t')
            if not fs[0] == 'en':
                continue
            synonyms[fs[1]].add(fs[3])
            synonyms[fs[3]].add(fs[1])
    return synonyms


def get_processer(args, sim_type='', vectors_fn=None):
    if not vectors_fn:
        vectors_fn = args.vectors
    if not sim_type:
        sim_type = args.sim_type
    batch = args.batch

    if sim_type in ("machine_only", "machine", "hybrid"):
        from pymachine.wrapper import Wrapper as MachineWrapper
        from pymachine.similarity import SentenceSimilarity as MachineSenSimilarity  # nopep8
        from pymachine.similarity import WordSimilarity as MachineWordSimilarity  # nopep8

    if sim_type == "machine_only":
        sts_wrapper = STSWrapper()
        machine_wrapper = MachineWrapper(
            'configs/machine.cfg', include_longman=True, batch=batch)
        machine_sim = MachineSenSimilarity(machine_wrapper)
        return lambda l: machine_sim.process_line(
            l, parser=sts_wrapper.parse_sts_line,
            sen_filter=sts_wrapper.filter_sen,
            fallback_sim=AlignAndPenalize.ngram_sim)

    wn_cache = WordnetCache()
    hunspell_wrapper = None
    #hunspell_wrapper = HunspellWrapper()

    if sim_type == 'lsa':
        lsa_wrapper = LSAWrapper()
        sts_wrapper = STSWrapper(sim_function=lsa_wrapper.word_similarity,
                                 wn_cache=wn_cache,
                                 hunspell_wrapper=hunspell_wrapper)
    elif sim_type == 'synonyms':
        synonyms = read_synonyms(args.synonyms)
        #logging.info('Synonyms read from file: {0}'.format(args.synonyms))
        sts_wrapper = STSWrapper(
            sim_function=lambda a, b, c, d: synonym_sim(synonyms, a, b),
            wn_cache=wn_cache, hunspell_wrapper=hunspell_wrapper)

    elif sim_type == 'machine':
        machine_wrapper = MachineWrapper(
            'configs/machine.cfg', include_longman=True, batch=batch)
        machine_sim = MachineWordSimilarity(machine_wrapper)
        sts_wrapper = STSWrapper(sim_function=machine_sim.word_similarity,
                                 wn_cache=wn_cache,
                                 hunspell_wrapper=hunspell_wrapper)

    elif sim_type == 'hybrid':
        lsa_wrapper = LSAWrapper()
        machine_wrapper = MachineWrapper(
            'configs/machine.cfg', include_longman=True, batch=batch)

        machine_sim = MachineWordSimilarity(machine_wrapper)

        hybrid_sim = HybridSimWrapper(lsa_wrapper, machine_sim)

        sts_wrapper = STSWrapper(sim_function=hybrid_sim.max_sim,
                                 wn_cache=wn_cache,
                                 hunspell_wrapper=hunspell_wrapper)

    elif sim_type == 'ngram':
        sts_wrapper = STSWrapper(sim_function=AlignAndPenalize.ngram_sim,
                                 wn_cache=wn_cache,
                                 hunspell_wrapper=hunspell_wrapper)

    elif sim_type == 'none':
        sts_wrapper = STSWrapper(sim_function=lambda a, b, c, d: None,
                                 wn_cache=wn_cache,
                                 hunspell_wrapper=hunspell_wrapper)

    elif sim_type == 'twitter_embedding':
        lsa_wrapper = LSAWrapper(vectors_fn, word2vec=False)
        sts_wrapper = STSWrapper(sim_function=lsa_wrapper.word_similarity,
                                 wn_cache=wn_cache,
                                 hunspell_wrapper=hunspell_wrapper)
    else:
        raise Exception('unknown similarity type: {0}'.format(sim_type))

    return sts_wrapper.process_line


args = parse_args()


def setup_full_test():
    processers = []
    for sim in ['ngram', 'machine', 'lsa']:
        processers.append(get_processer(args, sim_type=sim))
        logging.info('{0} initialized'.format(sim))
    for vec_fn in ['5gram/gensim_5gram',
                   '6gram_withhashtag/gensim_6gram_withhashtag']:
        processers.append(get_processer(
            args, sim_type='twitter_embedding', vectors_fn=vec_fn))
    return processers


def main():
    #sim_type = argv[1]
    #batch = len(argv) == 3 and argv[2] == 'batch'
    log_level = logging.WARNING if args.batch else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    logging.warning('Similarity type: {0}'.format(args.sim_type))
    if args.full_test:
        processers = setup_full_test()
        for c, line in enumerate(stdin):
            if c % 100 == 0:
                logging.warning('Processed {0} lines'.format(c))
            global feats
            all_feats = []
            if args.lower:
                line = line.lower()
            for pr in processers:
                feats = []
                sim = pr(line)
                all_feats.append(sim)
            all_feats.extend(feats[0])
            feat_f.write(' '.join(map(str, all_feats)) + '\n')
        return
    processer = get_processer(args)

    if not args.shell:
        for c, line in enumerate(stdin):
            if args.lower:
                line = line.lower()
            processer(line)
            if c % 100 == 0:
                logging.warning('{0}...'.format(c))
    else:
        logging.info('Initialization done')
        import readline
        assert readline  # silence pyflakes
        while(True):
            line1 = raw_input()
            line2 = raw_input()
            if args.lower:
                line1 = line1.lower()
                line2 = line2.lower()
            try:
                processer(line1 + '\t' + line2)
            except Exception as e:
                logging.exception(e)
                continue

if __name__ == '__main__':
    oov_stat = defaultdict(set)
    feat_f = open(args.features, 'w+')
    #import cProfile
    #cProfile.run('main()', 'log/stats.cprofile')
    main()
    if global_flags['log_oov_stat']:
        for k, v in oov_stat.iteritems():
            with open('log/tmp/' + k, 'w') as f:
                f.write('\n'.join(sorted(v)).encode('utf8'))
    feat_f.close()
