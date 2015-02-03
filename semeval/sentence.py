from collections import defaultdict


class Sentence(object):

    def __init__(self, sentence, tokens):
        self.sentence = sentence
        self.tokens = tokens
        self.add_compounds()
        self.add_acronyms()

    def add_compounds(self):
        self.compounds = {}
        for i, tok in enumerate(self.tokens[:-1]):
            self.compounds[u'{0}{1}'.format(tok['token'], self.tokens[i + 1]['token'])] = i
            self.compounds[u'{0}-{1}'.format(tok['token'], self.tokens[i + 1]['token'])] = i

    def add_acronyms(self):
        self.head_of = {}
        self.acronyms = defaultdict(set)
        for i in xrange(len(self.tokens) - 1):
            for j in range(2, 5):
                if i + j > len(self.tokens):
                    continue
                words = tuple([w['token'] for w in self.tokens[i:i + j]])
                abbr = ''.join(w[0] for w in words)
                self.acronyms[abbr].add((i, words))

    def __hash__(self):
        return hash(self.sentence)

    def __unicode__(self):
        #return u'{0} --> {1}\n  tags: {2}'.format(self.sentence, ' '.join(t['token'] for t in self.tokens), '\n  '.join(str(i) for i in self.tokens))
        return u'{0} --> {1}'.format(self.sentence, ' '.join(t['token'] for t in self.tokens))

    def __str__(self):
        return unicode(self).encode('utf8')


class SentencePair(object):

    def __init__(self, sen1, sen2):
        self.sen1 = sen1
        self.sen2 = sen2
        self.match1 = defaultdict(list)
        self.match2 = defaultdict(list)
        self.features = defaultdict(float)

    def __str__(self):
        return '{0}\n{1}\n'.format(self.sen1, self.sen2)
