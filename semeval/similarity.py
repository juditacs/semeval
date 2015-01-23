import logging


def get_similarity(config, section):
    sim_type = config.get(section, 'type')
    if sim_type == 'jaccard' or sim_type == 'dice':
        n = config.getint(section, 'ngram')
        padding = config.getboolean(section, 'padding')
        return NGramSimilarity(n, sim_type, padding)


class BaseSimilarity(object):

    def word_sim(self, word1, word2):
        logging.warning('The BaseSimilarity word_sim method was called. Did you forget to implement it in the derived class?')
        return 0.0


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
            sim = 0.0
        self.word_cache[(word1, word2)] = sim
        return sim
