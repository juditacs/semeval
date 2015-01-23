import re
import nltk
from resources import Resources


class SynsetWrapper(object):

    punct_re = re.compile(r'[\(\)]', re.UNICODE)

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
                    w.strip() and not w.strip() in Resources.stopwords)])
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

    def __init__(self, conf):
        self.conf = conf
        self.synsets = {}
        self.synset_to_wrapper = {}
        self.senses = {}

    def get_significant_synsets(self, word):
        if not word in self.synsets:
            candidates = nltk.corpus.wordnet.synsets(word)
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
        if self.conf.getboolean('wordnet', 'enrich_with_senses'):
            return set([word])
        if not word in self.senses:
            self.senses[word] = set([word])
            sn = nltk.corpus.wordnet.synsets(word)
            if len(sn) >= self.conf.getint('wordnet', 'sense_threshold'):
                th = len(sn) / 3.0
                for synset in sn:
                    for lemma in synset.lemmas():
                        lsn = nltk.corpus.wordnet.synsets(lemma.name())
                        if len(lsn) <= th:
                            self.senses[word].add(
                                lemma.name().replace('_', ' '))
        return self.senses[word]
