import logging
import math
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


class BaseSimilarity(object):

    def word_sim(self, word1, word2):
        logging.warning('The BaseSimilarity word_sim method was called. Did you forget to implement it in the derived class?')
        return 0.0


class LSASimilarity(BaseSimilarity):

    def __init__(self, section, config):
        self.model_fn = config.get(section, 'model')
        self.model_type = config.get(section, 'model_type')
        self.wordnet_boost = config.getboolean(section, 'wordnet_boost')
        self.use_twitter_norm = config.getboolean('alignment', 'twitter_norm')
        logging.info('Loading model: {0}'.format(self.model_fn))
        if self.model_type == 'word2vec':
            self.model = Word2Vec.load_word2vec_format(self.model_fn, binary=False)
        elif self.model_type == 'gensim':
            self.model = Word2Vec.load(self.model_fn)
        else:
            raise Exception('Unknown LSA model format')
        logging.info('Model loaded: {0}'.format(self.model_fn))
        self.sim_cache = {}
        self.lsa_sim_cache = {}

    def word_sim(self, word1, word2):
        if (word1, word2) in self.sim_cache:
            return self.sim_cache[(word1, word2)]
        cand1 = self.get_spell_variations(word1)
        cand2 = self.get_spell_variations(word2)
        max_pair = (word1, word2)
        max_sim = 0.0
        for c1 in cand1:
            for c2 in cand2:
                if c1 in self.model and c2 in self.model:
                    s = self.model.similarity(c1, c2)
                    #logging.info(u'Calling similarity: \t{0}\t{1}'.format(c1, c2).encode('utf8'))
                    if s > max_sim:
                        max_pair = (word1, word2)
                        max_sim = s
        if max_sim > 0.1:
            if self.wordnet_boost:
                D = Wordnet.get_boost(max_pair[0], max_pair[1])
                if not D is None:
                    max_sim += 0.5 * math.exp(-0.25 * D)
        else:
            max_sim = 0.0
        self.sim_cache[(word1, word2)] = max_sim
        self.sim_cache[(word2, word1)] = max_sim
        return max_sim

    def lsa_sim(self, word1, word2):
        if not (word1, word2) in self.lsa_sim_cache:
            try:
                s = self.model.similarity(word1, word2)
            except KeyError:
                s = None
            self.lsa_sim_cache[(word1, word2)] = s
            self.lsa_sim_cache[(word2, word1)] = s
        return self.lsa_sim_cache[(word1, word2)]

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
