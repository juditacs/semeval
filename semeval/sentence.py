from collections import defaultdict


class Sentence(object):

    def __init__(self, sentence, tokens):
        self.sentence = sentence
        self.tokens = tokens
        self.add_compounds()

    def add_compounds(self):
        self.compounds = set()
        for i, tok in enumerate(self.tokens[:-1]):
            self.compounds.add(u'{0}{1}'.format(tok['token'], self.tokens[i + 1]['token']))
            self.compounds.add(u'{0}-{1}'.format(tok['token'], self.tokens[i + 1]['token']))

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
