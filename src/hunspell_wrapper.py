import logging

class HunspellWrapper():

    @staticmethod
    def get_hunspell(prefix):
        try:
            from hunspell import HunSpell
            dic_fn = "{0}.dic".format(prefix)
            aff_fn = "{0}.dic".format(prefix)
            logging.info('loading hunspell dictionaries: {0} and {1}'.format(
                dic_fn, aff_fn))
            return HunSpell(dic_fn, aff_fn)
        except ImportError:
            logging.warning('hunspell is not present, using cache file only$')
            return None

    def __init__(self, aff_dic_prefix='/usr/share/hunspell/en_US',
                 cache_file='semeval_data/hunspell_cache'):
        self.hunspell = HunspellWrapper.get_hunspell(aff_dic_prefix)
        self.cache = {}
        self.read_cache_file(cache_file)

    def get_suggestions(self, word):
        if self.hunspell is None or ' ' in word:
            return []
        return self.hunspell.suggest(word)

    def read_cache_file(self, fn):
        for line in open(fn):
            fields = line.strip().split('\t')
            self.cache[fields[0]] = fields[1:]
