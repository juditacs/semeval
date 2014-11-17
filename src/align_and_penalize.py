from collections import defaultdict
import logging
import math
import os
import re
import string
from sys import stderr, stdin, argv

from gensim.models import Word2Vec


def log(s):
    stderr.write(s)
    stderr.flush()

from nltk.corpus import wordnet

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tag.hunpos import HunposTagger
from nltk.tokenize import word_tokenize

#from pymachine.src.wrapper import Wrapper as MachineWrapper

#assert MachineWrapper  # silence pyflakes

__EN_FREQ_PATH__ = '/mnt/store/home/hlt/Language/English/Freq/freqs.en'


class AlignAndPenalize(object):
    num_re = re.compile(r'^[0-9.,]+$', re.UNICODE)
    preferred_pos = ('VB', 'VBD', 'VBG', 'VBN', 'VBP',  # verbs
                     'NN', 'NNS', 'NNP', 'NNPS',   # nouns
                     'PRP', 'PRP$', 'CD')  # pronouns, numbers
    pronouns = {
        'i': 'me', 'me': 'i',
        'he': 'him', 'him': 'he',
        'she': 'her', 'her': 'she',
        'we': 'us', 'us': 'we',
        'they': 'them', 'them': 'they'}

    def __init__(self, sen1, sen2, tags1, tags2, map_tags, wrapper,
                 sim_function):
        logging.debug('AlignAndPenalize init:')
        logging.debug('sen1: {0}'.format(sen1))
        logging.debug('sen2: {0}'.format(sen2))
        self.sts_wrapper = wrapper
        if isinstance(sim_function, str):
            self.sim_function = getattr(self, sim_function)
        else:
            self.sim_function = sim_function
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

        self.sen1 = self.filter_sen(self.sen1)
        self.sen2 = self.filter_sen(self.sen2)

        self.compound_pairs = AlignAndPenalize.get_compound_pairs(self.sen1,
                                                                  self.sen2)

        logging.debug('compound pairs: {0}'.format(self.compound_pairs))

    @staticmethod
    def sts_map_tags(pos_tag, token_d):
        token_d['pos'] = pos_tag

    @staticmethod
    def twitter_map_tags(tags, token_d):
        sp = tags.split('/')[1:]
        token_d['ner'] = sp[0]
        token_d['pos'] = sp[1]
        token_d['chunk'] = sp[2]

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

    def filter_sen(self, sen):
        return [word for word in sen if (
            word['token'] not in self.sts_wrapper.punctuation and
            word['token'].lower() not in self.sts_wrapper.stopwords and
            not self.sts_wrapper.is_frequent_adverb(word['token'],
                                                    word['pos']))]

    def get_senses(self):
        for t in self.sen1 + self.sen2:
            if t['token'] in self.sts_wrapper.sense_cache:
                t['senses'] = self.sts_wrapper.sense_cache[t['token']]
                continue
            senses = set([t['token']])
            wn_sen = wordnet.synsets(t['token'])
            if len(wn_sen) >= 10:
                n = float(len(wn_sen))
                for s in wn_sen:
                    w = s.name().split('.')[0]
                    s_sen = wordnet.synsets(s.name())
                    if len(s_sen) / n <= 1.0 / 3:
                        senses.add(w)
            self.sts_wrapper.sense_cache[t['token']] = senses
            t['senses'] = senses

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
        logging.debug(u'got this: {0}, {1}'.format(x, y))
        for sx in self.sts_wrapper.sense_cache[x]:
            sim = self.similarity_wrapper(sx, y, x_pos, y_pos)
            if sim > max1:
                max1 = sim
                best_pair_1 = (sx, y)
        max2 = 0.0
        best_pair_2 = None
        for sy in self.sts_wrapper.sense_cache[y]:
            sim = self.similarity_wrapper(x, sy, x_pos, y_pos)
            if sim > max2:
                max2 = sim
                best_pair_2 = (sy, x)

        if max1 > max2:
            sim, best_pair = max1, best_pair_1
        else:
            sim, best_pair = max2, best_pair_2
        #logging.info('best pair: {0} ({1})'.format(sim, best_pair))
        return sim

    def baseline_similarity(self, x, y, x_i, y_i):
        return bigram_dist_jaccard(x, y)

    def similarity_wrapper(self, x, y, x_i, y_i):
        if self.is_num_equivalent(x, y):
            logging.info('equivalent numbers: {0}, {1}'.format(x, y))
            return 1
        if self.is_pronoun_equivalent(x, y):
            logging.info('equivalent pronouns: {0}, {1}'.format(x, y))
            return 1
        if self.is_acronym(x, y, x_i, y_i):
            logging.info('acronym match: {0}, {1}'.format(x, y))
            return 1
        if self.is_headof(x, y, x_i, y_i):
            logging.info('head_of match: {0}, {1}'.format(x, y))
            return 1
        if self.is_consecutive_match(x, y, x_i, y_i):
            logging.info('consecutive match: {0}, {1}'.format(x, y))
            return 1

        sim = self.sim_function(x, y, x_i, y_i)
        if sim is None:
            return self.bigram_sim(x, y)
        return sim

    def lsa_sim(self, x, y, x_i, y_i):
        #TODO
        if self.is_oov(x) or self.is_oov(y):
            return None
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
        if self.T > 1:
            raise Exception(
                'alignment score > 1: {0} {1}'.format(self.sen1, self.sen2))
        self.penalty()
        logging.info('T={0}, P={1}'.format(self.T, self.P))
        return self.T - self.P

    def weight_freq(self, token):
        if token in self.sts_wrapper.global_freqs:
            return 1 / self.sts_wrapper.global_freqs[token]
        return 1 / math.log(2)

    def weight_pos(self, pos):
        if pos in AlignAndPenalize.preferred_pos:
            logging.info('preferred pos: {0}'.format(pos))
            return 1
        logging.info('not preferred pos: {0}'.format(pos))
        return 0.5

    def is_antonym(self, w1, w2):
        if w1 in self.sts_wrapper.antonym_cache(w2):
            logging.info('Antonym found: {0} -- {1}'.format(w1, w2))
            return True
        if w2 in self.sts_wrapper.antonym_cache(w1):
            logging.info('Antonym found: {0} -- {1}'.format(w2, w1))
            return True
        return False

    def penalty(self):
        A1 = [t for t in self.sen1 if t['most_sim_score'] < 0.05]
        A2 = [t for t in self.sen2 if t['most_sim_score'] < 0.05]
        #logging.debug('A1: {0} words, A2: {1} words'.format(len(A1), len(A2)))
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
            logging.info('penalty for {0}: wf: {1}, wp: {2}'.format(
                (token, pos), self.weight_freq(token), self.weight_pos(pos)))
        P1A /= float(2 * len(self.sen1))
        P2A = 0.0
        for t in A2:
            score = t['most_sim_score']
            pos = t['pos']
            token = t['token']
            P2A += score + self.weight_freq(token) * self.weight_pos(pos)
            logging.info('penalty for {0}: wf: {1}, wp: {2}'.format(
                (token, pos), self.weight_freq(token), self.weight_pos(pos)))
        P2A /= float(2 * len(self.sen2))
        P1B = 0.0
        for t, gt, score in B1:
            P1B += score + 0.5
        P1B /= float(2 * len(self.sen1))
        P2B = 0.0
        for t, gt, score in B2:
            P2B += score + 0.5
        P2B /= float(2 * len(self.sen2))
        self.P = P1A + P2A + P1B + P2B
        logging.info('P1A: {0} P2A: {1} P1B: {2} P2B: {3}'.format(
            P1A, P2A, P1B, P2B))
        if self.P < 0:
            raise Exception(
                'negative penalty: {0}\n'.format(self.P) +
                'P1A: {0} P2A: {1} P1B: {2} P2B: {3}\n'.format(P1A, P2A, P1B,
                                                               P2B) +
                'sen1: {0}, sen2: {1}'.format(self.sen1, self.sen2))


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


class LSAWrapper(object):

    def __init__(self, vector_fn='vectors_example.bin'):
        self.lsa_model = Word2Vec.load_word2vec_format(os.path.join(os.environ['LSA_DIR'], vector_fn), binary=True)
        self.alpha = 0.25
        self.cache = {}
        self.wn_cache = WordnetCache()

    def wordnet_boost(self, word1, word2):
        sigsets1 = set(self.wn_cache.get_synsets(word1))
        sigsets2 = set(self.wn_cache.get_synsets(word2))
        #sigsets1 = self.significant_synset(word1)
        #sigsets2 = self.significant_synset(word2)
        if sigsets1 & sigsets2:
            # the two words appear together in the same synset
            return 0
        if self.is_direct_hypernym(sigsets1, sigsets2):
            logging.info('Direct hypernym: {0} -- {1} -- score: 1'.format(word1.encode('utf8'), word2.encode('utf8')))
            return 1
        if self.is_two_link_indirect_hypernym(sigsets1, sigsets2):
            return 2
        adj1 = set(filter(lambda x: x.pos() == 'a', sigsets1))
        adj2 = set(filter(lambda x: x.pos() == 'a', sigsets1))
        if adj1 and adj2:
            if self.is_direct_similar_to(adj1, adj2):
                logging.info('Direct similar to: {0} -- {1} -- score: 1'.format(word1.encode('utf8'), word2.encode('utf8')))
                return 1
            if self.is_two_link_indirect_similar_to(adj1, adj2):
                return 2
        if self.is_derivationally_related(sigsets1, sigsets2):
            logging.info('Derivationally rel: {0} -- {1} -- score: 1'.format(word1.encode('utf8'), word2.encode('utf8')))
            return 1
        #TODO
        # word is the head of the gloss of the other or its direct hypernym or one of its direct hyponyms
        # word appears frequently in the gloss of the other or its direct hypernym or one of its direct hyponyms
        # see Collins, 1999
        if self.in_hypopernym_glosses(word1, word2, sigsets1, sigsets2):
            return 2
        return None

    def in_hypopernym_glosses(self, word1, word2, synsets1, synsets2):
        if self.in_glosses(word1, synsets2):
            logging.info(u'Word [{0}] in glosses of word [{1}]'.format(word1, word2).encode('utf8'))
            return True
        if self.in_glosses(word2, synsets1):
            logging.info(u'Word [{0}] in glosses of word [{1}]'.format(word2, word1).encode('utf8'))
            return True

    def in_glosses(self, word, synsets):
        defs = defaultdict(int)
        for s in synsets:
            for word in s.definition():
                defs[word] += 1
            for h in s.hypernyms():
                for word in h.definition():
                    defs[word] += 1
            for h in s.hyponyms():
                for word in h.definition():
                    defs[word] += 1
        top3 = [i[0] for i in sorted(defs.iteritems(), key=lambda x: -x[1])[:5]]
        if word in top3:
            return True
        return False

    def is_derivationally_related(self, synsets1, synsets2):
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

    def significant_synset(self, word):
        synsets = wordnet.synsets(word)
        if len(synsets) == 0:
            logging.info('No synsets for word: {0}'.format(word.encode('utf8')))
            return set()
        sigsets = set([synsets[0]])
        for s in synsets[1:]:
            if self.wn_freq(s) >= 5:
                sigsets.add(s)
                continue
            sense_num = int(s.name().split('.')[-1])
            if s.lemmas()[0].name() == word and sense_num < 8:
                # I am no sure what they mean by Wordnet sense number
                sigsets.add(s)
                continue
        return sigsets

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
        try:
            self.lsa_model[word]
            return False
        except KeyError:
            return True

    def word_similarity(self, word1, word2, pos1, pos2):
        d = self.lookup_cache(word1, word2)
        if not d is None:
            return d
        if self.is_oov(word1) or self.is_oov(word2):
            return None
        sim = self.lsa_model.similarity(word1, word2)
        if sim < 0.1:
            logging.debug(u'LSA sim too low (less than 0.1), setting it to 0.0: {0} -- {1} -- {2}'.format(word1, word2, sim).encode('utf8'))
            return 0.0
        logging.info(u'LSA sim without wordnet: {0} -- {1} -- {2}'.format(word1, word2, sim).encode('utf8'))
        D = self.wordnet_boost(word1, word2)
        if D is not None:
            logging.info(u'LSA sim wordnet boost: {0} -- {1} -- {2}'.format(word1, word2, D).encode('utf8'))
            sim = sim + 0.5 * math.exp(-self.alpha * D)
        logging.info(u'LSA sim + wn boost: {0} -- {1} -- {2}'.format(word1, word2, sim).encode('utf8'))
        d = sim if sim <= 1 else 1
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
    nltk_sw = set(nltk_stopwords.words('english'))

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
            self._definition = set([w.strip() for w in def_.split() if w.strip() and not w.strip() in SynsetWrapper.nltk_sw])
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
            self._similar_tos = set(SynsetWrapper(s) for s in self.synset.similar_tos())
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

    def get_synsets(self, word):
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


class STSWrapper(object):

    custom_stopwords = set([])
    #custom_stopwords = set(["'s"])

    def __init__(self, sim_function='lsa_sim'):
        logging.info('reading global frequencies...')
        self.sim_function = sim_function
        self.read_freqs()
        self.sense_cache = {}
        self.frequent_adverbs_cache = {}
        self.punctuation = set(string.punctuation)
        self.hunpos_tagger = STSWrapper.get_hunpos_tagger()
        self.stopwords = STSWrapper.get_stopwords()
        self._antonym_cache = {}

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
        nltk_sw = set(nltk_stopwords.words('english'))
        return nltk_sw.union(STSWrapper.custom_stopwords)

    def parse_twitter_line(self, fd):
        sen1 = fd[2].split(' ')
        sen2 = fd[3].split(' ')
        tags1 = fd[5].split(' ')
        tags2 = fd[6].split(' ')
        return sen1, sen2, tags1, tags2

    def parse_sts_line(self, fields):
        sen1_toks, sen2_toks = map(word_tokenize, fields)
        try:
            sen1_pos, sen2_pos = map(
                lambda t: [tok[1] for tok in self.hunpos_tagger.tag(t)],
                (sen1_toks, sen2_toks))
        except UnicodeEncodeError:
            logging.warning('failed to run POS-tagging: {0}'.format(
                (sen1_toks, sen2_toks)))
            sen1_pos = ["UNKNOWN" for tok in sen1_toks]
            sen2_pos = ["UNKNOWN" for tok in sen2_toks]
        return sen1_toks, sen2_toks, sen1_pos, sen2_pos

    def read_freqs(self, ifn=__EN_FREQ_PATH__):
        self.global_freqs = {}
        with open(ifn) as f:
            for l in f:
                fd = l.decode('utf8').strip().split(' ')
                word = fd[0]
                logfreq = math.log(int(fd[1]) + 2)
                #we add 2 so that inverse logfreq makes sense for 0 and 1
                self.global_freqs[word] = logfreq

    def is_frequent_adverb(self, word, pos):
        return self.frequent_adverbs_cache.setdefault(
            (pos is not None and pos[:2] == 'RB' and
             self.global_freqs.get(word, 2) > 500000))

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
                                   wrapper=self,
                                   sim_function=self.sim_function)
        aligner.get_senses()
        aligner.get_most_similar_tokens()
        print aligner.sentence_similarity()


def main():
    batch = len(argv) == 2 and argv[1] == 'batch'
    log_level = logging.WARNING if batch else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    lsa_wrapper = LSAWrapper()
    sts_wrapper = STSWrapper(sim_function=lsa_wrapper.word_similarity)
    #sts_wrapper = STSWrapper(sim_function=machine_wrapper.word_similarity)
    #lsa_wrapper = LSAWrapper()
    #sts_wrapper = STSWrapper(sim_function=lsa_wrapper.word_similarity)

#    machine_wrapper = MachineWrapper(
#        os.path.join(os.environ['MACHINEPATH'],
#                     'pymachine/tst/definitions_test.cfg'),
#        include_longman=True, batch=batch)
    #sts_wrapper = STSWrapper(sim_function=machine_wrapper.word_similarity)
    #sts_wrapper = STSWrapper()
    for c, line in enumerate(stdin):
        sts_wrapper.process_line(line)
        if c % 100 == 0:
            logging.warning('{0}...'.format(c))

if __name__ == '__main__':
    main()
    #import cProfile
    #cProfile.run('main()', 'stats_new.cprofile')
