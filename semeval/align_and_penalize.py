from collections import defaultdict
from nltk.corpus import wordnet
import logging

from resources import Resources
import similarity


class AlignAndPenalize(object):

    word_cache = {}

    def __init__(self, conf):
        self.conf = conf
        self.fallback_similarity = None
        self.init_similarities()
        self._antonym_cache = {}

    def init_similarities(self):
        self.similarities = {}
        for section in self.conf.sections():
            if section.startswith('similarity_'):
                sim_name = section[11:]
                self.similarities[sim_name] = similarity.get_similarity(self.conf, section)
            elif 'fallback_' in section:
                self.fallback_similarity = similarity.get_similarity(self.conf, section)

    def align(self, pair):
        self.align_src_tgt(pair.match1, pair.sen1, pair.sen2)
        self.align_src_tgt(pair.match2, pair.sen2, pair.sen1)
        # update with context-based info
        self.update_with_context(pair)
        # comput P and T scores
        self.sentence_similarity(pair)
        final_score = self.compute_final_score(pair)
        return final_score

    def compute_penalty(self, pair):
        penalties = []
        if self.conf.getboolean('penalty', 'sim_too_low'):
            penalties.append(self.penalize_low_sim(pair))
        if self.conf.getboolean('penalty', 'penalize_antonyms'):
            penalties.extend(self.penalize_antonyms(pair))
        if self.conf.getboolean('penalty', 'penalize_named_entities'):
            penalties.append(self.ne_penalty(pair))
        if self.conf.getboolean('penalty', 'penalize_questions'):
            penalties.append(self.question_penalty(pair))
        if self.conf.getboolean('penalty', 'penalize_verb_tense'):
            penalties.append(self.verb_tense_penalty(pair))
        return self.summarize_penalties(penalties)

    def summarize_penalties(self, penalties):
        sum_pen = defaultdict(lambda: 0.0)
        for p in penalties:
            for typ, sc in p.iteritems():
                sum_pen[typ] += sc
        return sum_pen

    def question_penalty(self, pair):
        isq1 = (pair.sen1.tokens[0]['token'].lower() in
                Resources.question_starters)
        isq2 = (pair.sen2.tokens[0]['token'].lower() in
                Resources.question_starters)
        if isq1 == isq2:
            return defaultdict(lambda: 0)
        return defaultdict(lambda: 1.0 / (len(pair.sen1.tokens) + len(pair.sen2.tokens)))

    def verb_tense_penalty(self, pair):
        past = set(['vbd', 'vbn'])
        is_past1 = False
        for tok in pair.sen1.tokens:
            if tok['pos'].lower() in past:
                is_past1 = True
        is_past2 = False
        for tok in pair.sen2.tokens:
            if tok['pos'].lower() in past:
                is_past2 = True
        if is_past1 == is_past2:
            pair.features['PE'] = 0
            return defaultdict(lambda: 0)
        pair.features['PE'] = 1.0 / (len(pair.sen1.tokens) + len(pair.sen2.tokens))
        return defaultdict(lambda: 1.0 / (len(pair.sen1.tokens) + len(pair.sen2.tokens)))

    def penalize_antonyms(self, pair):
        b1 = defaultdict(float)
        b2 = defaultdict(float)
        for typ, match in pair.match1.iteritems():
            for i, (sc, t2) in enumerate(match):
                if self.is_antonym(pair.sen1.tokens[i]['token'], pair.sen2.tokens[t2]['token']):
                    b1[typ] += sc + 0.5
        for typ, match in pair.match2.iteritems():
            for i, (sc, t2) in enumerate(match):
                if self.is_antonym(pair.sen2.tokens[i]['token'], pair.sen1.tokens[t2]['token']):
                    b1[typ] += sc + 0.5
        pair.features['P1B'] = b1
        pair.features['P2B'] = b2
        return b1, b2

    def penalize_low_sim(self, pair):
        th = 0.05
        PA = {}
        for typ, match in pair.match1.iteritems():
            p1 = sum(m[0] for m in match if m[0] < th) / len(pair.sen1.tokens)
            p2 = sum(m[0] for m in pair.match2[typ] if m[0] < th) / len(pair.sen2.tokens)
            PA[typ] = (p1 + p2) / 2
            pair.features['P1A_' + typ] = p1
            pair.features['P2A_' + typ] = p2
        return PA

    def compute_final_score(self, pair):
        mode = self.conf.get('final_score', 'mode')
        if mode == 'average':
            score = sum((s[0] + s[1]) for s in pair.T.itervalues()) / len(pair.T)
        elif mode == 'max':
            score = max(s[0] + s[1] for s in pair.T.itervalues())
        elif mode == 'min':
            score = min(s[0] + s[1] for s in pair.T.itervalues())
        elif mode.startswith('similarity_'):
            typ = mode[11:]
            score = (pair.T[typ][0] + pair.T[typ][1])
        elif mode == 'regression':
            for typ, (sc1, sc2) in pair.T.iteritems():
                pair.features[typ] = sc1 + sc2
            return None
        return score

    def sentence_similarity(self, pair):
        pair.T = {}
        #self.T = defaultdict(lambda: [0.0, 0.0])
        pair.P = self.compute_penalty(pair)
        for typ, scores in pair.match1.iteritems():
            pair.T[typ] = [0, 0]
            pair.T[typ][0] = sum(s[0] for s in scores) / float(2 * len(scores)) - pair.P[typ] / 2
        for typ, scores in pair.match2.iteritems():
            pair.T[typ][1] = sum(s[0] for s in scores) / float(2 * len(scores)) - pair.P[typ] / 2

    def align_src_tgt(self, match, sen1, sen2):
        for typ in self.similarities.iterkeys():
            match[typ] = []
        for i, tok1 in enumerate(sen1.tokens):
            scores, best_toks = self.best_alignment(tok1, sen2.tokens)
            for typ, sc in scores.iteritems():
                match[typ].append((sc, best_toks[typ]))

    def update_with_context(self, pair):
        if self.conf.getboolean('alignment', 'compound_match'):
            self.update_with_compound_match(pair)
        if self.conf.getboolean('alignment', 'acronym_match'):
            self.update_with_acronym_scores(pair)

    def update_with_compound_match(self, pair):
        match1 = pair.match1
        match2 = pair.match2
        sen1 = pair.sen1
        sen2 = pair.sen2
        self.update_one_way_compound(sen1, sen2, match1, match2)
        self.update_one_way_compound(sen2, sen1, match2, match1)

    def update_one_way_compound(self, sen1, sen2, match1, match2):
        for i, t in enumerate(sen1.tokens):
            if t['token'] in sen2.compounds:
                for typ, v in match1.iteritems():
                    comp_i = sen2.compounds[t['token']]
                    v[i] = (1.0, comp_i)
                    match2[typ][comp_i] = (1.0, i)
                    match2[typ][comp_i + 1] = (1.0, i)

    def update_with_acronym_scores(self, pair):
        match1 = pair.match1
        match2 = pair.match2
        sen1 = pair.sen1
        sen2 = pair.sen2
        self.update_one_way_acronym(sen1, sen2, match1, match2)
        self.update_one_way_acronym(sen2, sen1, match2, match1)

    def update_one_way_acronym(self, sen1, sen2, match1, match2):
        for i, t in enumerate(sen1.tokens):
            if t['token'] in sen2.acronyms:
                for typ, v in match1.iteritems():
                    m = iter(sen2.acronyms[t['token']]).next()
                    v[i] = (1.0, m[0])
                    for head_i in range(m[0], m[0] + len(t['token'])):
                        match2[typ][head_i] = (1.0, i)
                    v[i] = (1.0, m[0])

    def best_alignment(self, left, right_tokens):
        max_sim = defaultdict(lambda: 0.0)
        max_j = defaultdict(lambda: -1)
        for j, rt in enumerate(right_tokens):
            for typ, simtype in self.similarities.iteritems():
                sim = self.senses_sim(left, rt, typ)
                if sim >= max_sim[typ]:
                    max_sim[typ] = sim
                    max_j[typ] = j
        return max_sim, max_j

    def senses_sim(self, word1, word2, simtype):
        max_sim = -1.0
        for w1 in word1['senses']:
            for w2 in word2['senses']:
                s = self.word_sim(w1, w2, simtype)
                #print(u'SENSES sim: {0}->{1} -- {2}->{3}: {4}'.format(word1['token'], w1, word2['token'], w2, s).encode('utf8'))
                if s > max_sim:
                    max_sim = s
        return max_sim

    def word_sim(self, w1, w2, simtype):
        if (w1, w2) in AlignAndPenalize.word_cache:
            return AlignAndPenalize.word_cache[(w1, w2)]
        # context-free similarity
        if w1 == w2 or Resources.is_num_equivalent(w1, w2) or Resources.is_pronoun_equivalent(w1, w2):
            AlignAndPenalize.word_cache[(w1, w2)] = 1
            AlignAndPenalize.word_cache[(w2, w1)] = 1
            return 1
        sim = self.similarities[simtype].word_sim(w1, w2)
        if sim is None and self.fallback_similarity:
            sim = self.fallback_similarity.word_sim(w1, w2)
        elif sim is None:
            sim = 0.0
        return sim

    def antonym_cache(self, key):
        #TODO use wordnet_cache
        if not key in self._antonym_cache:
            self._antonym_cache[key] = set()
            for synset in wordnet.synsets(key):
                for lemma in synset.lemmas():
                    for antonym in lemma.antonyms():
                        self._antonym_cache[key].add(
                            antonym.name().split('.')[0])
        return self._antonym_cache[key]

    def is_antonym(self, w1, w2):
        if w1 in self.antonym_cache(w2):
            logging.info('Antonym found: {0} -- {1}'.format(w1, w2))
            return True
        if w2 in self.antonym_cache(w1):
            return True
        return False

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

    def ne_penalty(self, pair):
        ne1, ne2 = self.collect_entities(pair)
        if not ne1 and not ne2:
            return defaultdict(float)
        match1, missing1 = self.find_ne_in_other(ne1, ne2)
        match2, missing2 = self.find_ne_in_other(ne2, ne1)
        pair.features['match1'] = len(match1)
        pair.features['match2'] = len(match2)
        pair.features['missing1'] = len(missing1)
        pair.features['missing2'] = len(missing2)
        if not match1 and not match2:
            pair.features['PC'] = 1
            return defaultdict(lambda: 1)
        diff1 = float(len(match1 - match2)) / len(match1 | match2)
        diff2 = float(len(match2 - match1)) / len(match1 | match2)
        pair.features['PC'] = 1 - max((diff1, diff2))
        return defaultdict(lambda: 1 - max([diff1, diff2]))

    def collect_entities(self, pair):
        current_ne = []
        typ = ''
        ne1 = defaultdict(list)
        ne2 = defaultdict(list)
        for tok in pair.sen1.tokens:
            tok['ner'] = tok['ner'].lower()
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
        for tok in pair.sen2.tokens:
            tok['ner'] = tok['ner'].lower()
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
