import os
from HTMLParser import HTMLParser
from ConfigParser import NoOptionError
import nltk

from sentence import Sentence
from resources import Resources
from wordnet_cache import WordnetCache as Wordnet


class ReadAndEnrich(object):

    def __init__(self, conf):
        self.conf = conf
        self.enricher = Enricher(conf)
        self.pairs = []

    def read_sentences(self, stream):
        for sen1, sen2, tags1, tags2 in self.read_lines(stream):
            s1 = self.enricher.add_sentence(sen1, tags1)
            s2 = self.enricher.add_sentence(sen2, tags2)
            self.pairs.append((s1, s2))
        return self.pairs

    def clear_pairs(self):
        self.pairs = []

    def read_lines(self, stream):
        enc = self.conf.get('global', 'encoding')
        for l in stream:
            fs = l.decode(enc).strip().split('\t')
            if len(fs) == 2:
                sen1 = fs[0]
                sen2 = fs[1]
                yield sen1, sen2, None, None
            elif len(fs) == 6:
                sen1 = fs[2]
                sen2 = fs[3]
                tags1 = fs[4]
                tags2 = fs[5]
                yield sen1, sen2, tags1, tags2
            elif len(fs) == 7:
                sen1 = fs[2]
                sen2 = fs[3]
                tags1 = fs[5]
                tags2 = fs[6]
                yield sen1, sen2, tags1, tags2


class Enricher(object):

    def __init__(self, conf):
        self.conf = conf
        self.sentences = {}
        if self.conf.get('global', 'tokenizer') == 'sts':
            self.html_parser = HTMLParser()
            self.hunpos = self.init_hunpos()

    def init_hunpos(self):
        try:
            hunpos_dir = self.conf.get('global', 'hunpos_dir')
        except NoOptionError:
            hunpos_dir = '/home/recski/projects/hundisambig_compact'
        hunpos_binary = os.path.join(hunpos_dir, 'hunpos-tag')
        hunpos_model = os.path.join(hunpos_dir, 'en_wsj.model')
        return nltk.tag.HunposTagger(hunpos_model, hunpos_binary)

    def add_sentence(self, sentence, tags):
        if not sentence in self.sentences:
            tokens = self.tokenize_and_tag(sentence, tags)
            self.add_wordnet_senses(tokens)
            # filter tokens if the config option remove_stopwords
            # and/or remove_punctuation is set
            filt_tokens = self.filter_tokens(tokens)
            self.add_wordnet_senses(filt_tokens)
            s = Sentence(sentence, filt_tokens)
            s.orig_tokens = [t for t in tokens]
            self.sentences[hash(s)] = s
        return self.sentences[hash(sentence)]

    def add_wordnet_senses(self, tokens):
        for token in tokens:
            if self.conf.getboolean('wordnet', 'enrich_with_senses'):
                token['senses'] = Wordnet.get_senses(token['token'], self.conf.getint('wordnet', 'sense_threshold'))
            else:
                token['senses'] = set([token['token']])

    def filter_tokens(self, tokens):
        new_tok = []
        for token in tokens:
            word = token['token']
            if self.conf.getboolean('global', 'remove_stopwords') and word in Resources.stopwords:
                continue
            if self.conf.getboolean('global', 'remove_punctuation') and word in Resources.punctuation:
                continue
            if self.conf.getboolean('global', 'filter_frequent_adverbs') and Resources.is_frequent_adverb(word, token['pos']):
                continue
            new_tok.append(token)
        return new_tok

    def tokenize_and_tag(self, sentence, tags):
        tokens = [{'token': t} for t in self.tokenize(sentence)]
        if tags:
            self.parse_tags(tokens, tags)
        else:
            if self.conf.get('global', 'tokenizer') == 'sts':
                self.tag_tokens(tokens)
            else:
                self.dummy_tag_tokens(tokens)
        if self.conf.getboolean('global', 'lower'):
            for t in tokens:
                t['token'] = t['token'].lower()
        return tokens

    def tokenize(self, sentence):
        tok_method = self.conf.get('global', 'tokenizer')
        if tok_method == 'simple':
            return sentence.split(' ')
        if tok_method == 'sts':
            return self.sts_tokenize(sentence)

    def sts_tokenize(self, sentence):
        tokens = nltk.word_tokenize(self.html_parser.unescape(sentence))
        toks = []
        for tok in tokens:
            if tok in Resources.punctuation:
                toks.append(tok)
            else:
                toks += Resources.punct_re.split(tok)
        return filter(lambda x: x not in ('', 's'), toks)

    def parse_tags(self, tokens, tags_str):
        # match tags with tokens and skip tags if a token
        # is missing (it was filtered by the tokenizer)
        i = 0
        for t in tags_str.split():
            sp = t.split('/')
            if not sp[0] == tokens[i]['token']:
                continue
            tokens[i]['ner'] = sp[1]
            tokens[i]['pos'] = sp[2]
            tokens[i]['chunk'] = sp[3]
            i += 1

    def dummy_tag_tokens(self, tokens):
        for t in tokens:
            t['pos'] = ''
            t['ner'] = ''
            t['chunk'] = ''

    def tag_tokens(self, tokens):
        words = [i['token'] for i in tokens]
        pos_tags = self.hunpos.tag(words)
        if self.conf.getboolean('penalty', 'penalize_named_entities'):
            ne = nltk.ne_chunk(pos_tags)
            ner_tags = self.get_ner_tags(ne)
        else:
            ner_tags = ['' for _ in range(len(tokens))]
        for i, tag in enumerate(pos_tags):
            tokens[i]['pos'] = tag
            tokens[i]['ner'] = ner_tags[i]

    def get_ner_tags(self, ne):
        tags = []
        for piece in ne:
            if isinstance(piece, tuple):
                tok, pos = piece
                tags.append((pos, 'o'))
            else:
                ne_type = piece.label()
                tags.append((piece[0][1], 'b-{0}'.format(ne_type)))
                tags += [(tok[1], 'i-{0}'.format(ne_type)) for tok in piece[1:]]
        return tags
