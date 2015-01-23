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
        self.T = self.sentence_similarity(pair)
        #TODO penalty
        final_score = self.compute_final_score()
        return final_score

    def compute_final_score(self):
        mode = self.conf.get('final_score', 'mode')
        if mode == 'average':
            score = sum((s[0] + s[1]) / 2.0 for s in self.T.itervalues()) / len(self.T)
            return score
        elif mode == 'max':
            return max(s[0] + s[1] for s in self.T.itervalues()) / 2.0
        elif mode == 'min':
            return min(s[0] + s[1] for s in self.T.itervalues()) / 2.0
        elif mode.startswith('similarity_'):
            typ = mode.lstrip('similarity_')
            return (self.T[typ][0] + self.T[typ][1]) / 2.0

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

    def best_alignment(self, left, right_tokens):
        max_sim = defaultdict(lambda: -1.0)
        max_j = defaultdict(lambda: -1)
        for j, rt in enumerate(right_tokens):
            for typ, simtype in self.similarities.iteritems():
                sim = self.senses_sim(left, rt, typ)
                if sim > max_sim[typ]:
                    max_sim[typ] = sim
                    max_j[typ] = j
        return max_sim, max_j

    def senses_sim(self, word1, word2, simtype):
        max_sim = 0.0
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
