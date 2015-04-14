import logging
import random
import math
from collections import defaultdict
from gensim.models import Word2Vec

from resources import Resources
from wordnet_cache import WordnetCache as Wordnet


def get_similarity(config, section):
    sim_type = config.get(section, 'type')
    if sim_type == 'jaccard' or sim_type == 'dice':
        n = config.getint(section, 'ngram')
        padding = config.getboolean(section, 'padding')
        return NGramSimilarity(n, sim_type, padding)
    if sim_type == 'lsa':
        return LSASimilarity(section, config)
    if sim_type == 'machine':
        return MachineSimilarity(config)
    if sim_type == 'synonyms':
        syn_fn = config.get(section, 'synonyms_file')
        tolower = config.getboolean('global', 'lower')
        return SynonymSimilarity(syn_fn, tolower)
    if sim_type == 'random':
        return RandomSimilarity()
    if sim_type == 'none':
        return NoneSimilarity()
    raise Exception('Unknown similarity: {0}'.format(section))


class BaseSimilarity(object):

    def word_sim(self, word1, word2):
        logging.warning('The BaseSimilarity word_sim method was called. Did you forget to implement it in the derived class?')  # nopep8
        return 0.0


class LSASimilarity(BaseSimilarity):

    def __init__(self, section, config):
        self.model_fn = config.get(section, 'model')
        self.model_type = config.get(section, 'model_type')
        self.wordnet_boost = config.getboolean(section, 'wordnet_boost')
        self.use_twitter_norm = config.getboolean('alignment', 'twitter_norm')
        logging.info('Loading model: {0}'.format(self.model_fn))
        if self.model_type == 'word2vec':
            self.model = Word2Vec.load_word2vec_format(
                self.model_fn, binary=False)
        elif self.model_type == 'gensim':
            self.model = Word2Vec.load(self.model_fn)
        else:
            raise Exception('Unknown LSA model format')
        logging.info('Model loaded: {0}'.format(self.model_fn))
        self.sim_cache = {}

    def word_sim(self, word1, word2):
        if (word1, word2) in self.sim_cache:
            return self.sim_cache[(word1, word2)]
        cand1 = self.get_spell_variations(word1)
        cand2 = self.get_spell_variations(word2)
        max_pair = (word1, word2)
        max_sim = None
        for c1 in cand1:
            for c2 in cand2:
                if c1 in self.model and c2 in self.model:
                    s = self.model.similarity(c1, c2)
                    # logging.info(
                    #     u'Calling similarity: \t{0}\t{1}'.format(
                    #         c1, c2).encode('utf8'))
                    if not max_sim or s > max_sim:
                        max_pair = (word1, word2)
                        max_sim = s
        if max_sim is not None:
            if max_sim > 0.1:
                if self.wordnet_boost:
                    D = Wordnet.get_boost(max_pair[0], max_pair[1])
                    if D is not None:
                        max_sim += 0.5 * math.exp(-0.25 * D)
            else:
                max_sim = 0.0
        if max_sim is not None and max_sim > 1.0:
            max_sim = 1.0
        self.sim_cache[(word1, word2)] = max_sim
        self.sim_cache[(word2, word1)] = max_sim
        return max_sim

    def get_spell_variations(self, word):
        variations = set([word])
        if self.use_twitter_norm:
            variations |= Resources.twitter_candidates(word, self.model)
        return variations


class NGramSimilarity(BaseSimilarity):

    def __init__(self, n, sim_type='jaccard', padding=False):
        self.n = n
        self.padding = padding
        self.sim_type = sim_type
        self.word_cache = {}

    def get_ngrams(self, word):
        if self.padding:
            word = '*' * (self.n - 1) + word + '*' * (self.n - 1)
        ngrams = set()
        for i in xrange(len(word) - self.n + 1):
            ngrams.add(word[i:i + self.n])
        return ngrams

    def sim_func(self, ng1, ng2):
        if self.sim_type == 'jaccard':
            return float(len(ng1 & ng2)) / len(ng1 | ng2)
        if self.sim_type == 'dice':
            return float(2 * len(ng1 & ng2)) / (len(ng1) + len(ng2))

    def word_sim(self, word1, word2):
        if (word1, word2) in self.word_cache:
            return self.word_cache[(word1, word2)]
        if (word2, word1) in self.word_cache:
            return self.word_cache[(word2, word1)]
        ng1 = self.get_ngrams(word1)
        ng2 = self.get_ngrams(word2)
        try:
            sim = self.sim_func(ng1, ng2)
        except ZeroDivisionError:
            sim = None
        self.word_cache[(word1, word2)] = sim
        return sim


class MachineSimilarity(BaseSimilarity):

    def __init__(self, cfg):
        from fourlang.similarity import WordSimilarity as FourlangWordSimilarity  # nopep8
        self.fourlang_sim = FourlangWordSimilarity(cfg)

    def word_sim(self, word1, word2):
        return self.fourlang_sim.word_similarity(word1, word2, -1, -1)


class SynonymSimilarity(BaseSimilarity):

    def __init__(self, syn_fn, tolower=False):
        self.synonyms = defaultdict(set)
        if syn_fn.endswith('.gz') or syn_fn.endswith('.gzip'):
            import gzip
            stream = gzip.open(syn_fn)
        else:
            stream = open(syn_fn)
        for l in stream:
            if tolower:
                fs = l.decode('utf8').lower().strip().split('\t')
            else:
                fs = l.decode('utf8').strip().split('\t')
            if not fs[0] == 'en':
                continue
            self.synonyms[fs[1]].add(fs[3])
            self.synonyms[fs[3]].add(fs[1])
        logging.info('Synonyms read from: {0}'.format(syn_fn))

    def word_sim(self, word1, word2):
        if word1 not in self.synonyms or word2 not in self.synonyms:
            return None
        if word2 in self.synonyms[word1]:
            return 1.0
        return 0.0


class RandomSimilarity(BaseSimilarity):

    def word_sim(self, word1, word2):
        return random.random()


class NoneSimilarity(BaseSimilarity):

    def word_sim(self, word1, word2):
        return None
