#!/usr/bin/env python

import logging
import os
import sys

from nltk.tokenize import word_tokenize

from hunmisc.utils.huntool_wrapper import Hundisambig, Ocamorph, MorphAnalyzer
from pymachine.src.machine import MachineGraph
from pymachine.src.wrapper import Wrapper

__USAGE__ = 'usage:\n' + \
    "machine_sim.py sen_file dep_file cfg_file (two sentences)" + \
    'machine_sim.py sen_dir dep_dir cfg_file (batch evaluation)'

def get_deps(stream):
    sen1, sen2 = [], []
    while True:
        dep = stream.readline().strip()
        if not dep:
            break
        sen1.append(dep)
    for line in stream:
        dep = line.strip()
        if not dep:
            break
        sen2.append(dep)

    return sen1, sen2

def get_sim(graph1, graph2):
    all_edges = graph1.edges.union(graph2.edges)
    logging.debug('all edges: {}'.format(all_edges))
    shared_edges = graph1.edges.intersection(graph2.edges)
    logging.debug('shared edges: {}'.format(shared_edges))
    return len(shared_edges) / float(len(all_edges))

def get_graph(deps, wrapper, tok2lemma):
    wrapper.reset_lexicon()
    for binary in wrapper.lexicon.active_machines():
        wrapper.wordlist.add(binary.printname())
    for dep in deps:
        wrapper.add_dependency(dep, tok2lemma)
    active_machines = wrapper.lexicon.active_machines()
    logging.debug('active machines: {}'.format(active_machines))
    return MachineGraph.create_from_machines(
        active_machines, max_depth=2, whitelist=wrapper.wordlist)

def main(sens_path, deps_path, cfg_file):

    hunmorph_dir = os.environ['HUNMORPH_DIR']
    analyzer = MorphAnalyzer(
        Ocamorph(
            os.path.join(hunmorph_dir, "ocamorph"),
            os.path.join(hunmorph_dir, "morphdb_en.bin")),
        Hundisambig(
            os.path.join(hunmorph_dir, "hundisambig"),
            os.path.join(hunmorph_dir, "en_wsj.model")))

    tok2lemma = {"ROOT": "ROOT"}

    w = Wrapper(cfg_file)

    if os.path.isdir(sens_path):
        assert os.path.isdir(deps_path), __USAGE__
        gold_path = os.path.join(os.environ['SEMEVAL_DATA'], 'sts_trial')
        assert os.path.isdir(gold_path), "{} does not exist".format(gold_path)

    else:
        assert not os.path.isdir(deps_path), __USAGE__
        sen1_toks, sen2_toks = map(word_tokenize, open(sens_path).readlines())
        sen1_ana, sen2_ana = analyzer.analyze([sen1_toks, sen2_toks])
        for tok, ana in sen1_ana + sen2_ana:
            tok2lemma[tok] = ana.split('||')[0].split('<')[0]

        sen1_deps, sen2_deps = get_deps(file(deps_path))

        graphs = (get_graph(sen1_deps, w, tok2lemma),
                  get_graph(sen2_deps, w, tok2lemma))
        for c, graph in enumerate(graphs):
            f = open('sen{}.dot'.format(c), 'w')
            f.write(graphs[c].to_dot())
            f.close()

        sim = get_sim(graphs[0], graphs[1])
        print 'similarity: {}'.format(sim)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    assert len(sys.argv) == 4, __USAGE__
    sens_path, deps_path, cfg_file = sys.argv[1:4]
    main(sens_path, deps_path, cfg_file)
