from collections import defaultdict

from resources import Resources
import similarity


class AlignAndPenalize(object):

    word_cache = {}

    def __init__(self, conf):
        self.conf = conf
        self.init_similarities()
        self.fallback_similarity = None

    def init_similarities(self):
        self.similarities = {}
        for section in self.conf.sections():
            if not section.startswith('similarity_'):
                if section == 'fallback_similarity':
                    self.fallback_similarity = similarity.get_similarity(self.conf, section)
                continue
            sim_name = section.lstrip('similarity_')
            self.similarities[sim_name] = similarity.get_similarity(self.conf, section)

    def align(self, pair):
        self.align_src_tgt(pair.match1, pair.sen1, pair.sen2)
        self.align_src_tgt(pair.match2, pair.sen2, pair.sen1)
        # update with context-based info
        self.update_with_context(pair)
#        print('MATCH1')
#        for i, (sc, j) in enumerate(pair.match1['dice4']):
#            print(u'{0} -> {1}: {2}'.format(pair.sen1.tokens[i]['token'], pair.sen2.tokens[j]['token'], sc).encode('utf8'))
#        print('MATCH2')
#        for i, (sc, j) in enumerate(pair.match2['dice4']):
#            print(u'{0} -> {1}: {2}'.format(pair.sen2.tokens[i]['token'], pair.sen1.tokens[j]['token'], sc).encode('utf8'))
#        print pair.match1
        self.T = self.sentence_similarity(pair)
        self.P = self.compute_penalty(pair)
        final_score = self.compute_final_score()
        return final_score

    def compute_penalty(self, pair):
        if self.conf.getboolean('penalty', 'sim_too_low'):
            PA = self.penalize_low_sim(pair)
        else:
            PA = 0
        #TODO
        return PA

    def penalize_low_sim(self, pair):
        #TODO
        return 0

    def compute_final_score(self):
        mode = self.conf.get('final_score', 'mode')
        if mode == 'average':
            score = sum((s[0] + s[1]) for s in self.T.itervalues()) / len(self.T)
            return score
        elif mode == 'max':
            return max(s[0] + s[1] for s in self.T.itervalues())
        elif mode == 'min':
            return min(s[0] + s[1] for s in self.T.itervalues())
        elif mode.startswith('similarity_'):
            typ = mode.lstrip('similarity_')
            return (self.T[typ][0] + self.T[typ][1])

    def sentence_similarity(self, pair):
        sum_scores = defaultdict(lambda: [0.0, 0.0])
        for typ, scores in pair.match1.iteritems():
            sum_scores[typ][0] = sum(s[0] for s in scores) / float(2 * len(scores))
        for typ, scores in pair.match2.iteritems():
            sum_scores[typ][1] = sum(s[0] for s in scores) / float(2 * len(scores))
        return sum_scores

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
                if sim > max_sim[typ]:
                    max_sim[typ] = sim
                    max_j[typ] = j
        return max_sim, max_j

    def senses_sim(self, word1, word2, simtype):
        max_sim = -1.0
        for w1 in word1['senses']:
            for w2 in word2['senses']:
                s = self.word_sim(w1, w2, simtype)
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
        #TODO acronym
        #TODO headof
        #TODO compound
        sim = self.similarities[simtype].word_sim(w1, w2)
        if sim is None and self.fallback_similarity:
            return self.fallback_similarity.word_sim(w1, w2)
        return sim
