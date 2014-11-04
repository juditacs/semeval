#!/usr/bin/env python

import logging
import os
import sys

from nltk.tokenize import word_tokenize

from pymachine.src.machine import MachineGraph
from pymachine.src.wrapper import Wrapper

__USAGE__ = 'usage:\n' + \
    "machine_sim.py sen_file dep_file cfg_file (two sentences)" + \
    'machine_sim.py sen_dir dep_dir out_dir cfg_file (batch evaluation)'

def print_now(s, newline=True):
    if newline:
        print s
    else:
        print s,

    sys.stdout.flush()

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

def graph_sim(graph1, graph2):
    all_edges = graph1.edges.union(graph2.edges)
    logging.debug('all edges: {}'.format(all_edges))
    shared_edges = graph1.edges.intersection(graph2.edges)
    logging.debug('shared edges: {}'.format(shared_edges))
    info_str = "graph1_edges: {}, graph2_edges: {}, shared: {}".format(
        len(graph1.edges), len(graph2.edges), len(shared_edges))
    return len(shared_edges) / float(len(all_edges)), info_str

def get_graph(deps, whitelist_toks, wrapper):
    print_now('resetting lexicon...')
    wrapper.reset_lexicon()
    wrapper.wordlist = set()
    for binary in wrapper.lexicon.active_machines():
        wrapper.wordlist.add(binary.printname())
    print_now('adding dependencies...')
    for dep in deps:
        wrapper.add_dependency(dep)
    active_machines = wrapper.lexicon.active_machines()
    logging.debug('active machines: {}'.format(active_machines))

    wordlist = set([wrapper.get_lemma(tok) for tok in whitelist_toks])
    wordlist_machines = map(wrapper.lexicon.get_machine, wordlist)
    print_now('creating graph...')
    print_now('whitelist: {0}'.format(wordlist))
    return MachineGraph.create_from_machines(
        wordlist_machines, max_depth=0, whitelist=wordlist, strict=True)
    #return MachineGraph.create_from_machines(
    #    wordlist_machines, max_depth=1)

def get_sim(sen_file, dep_file, wrapper, to_dot=False):
        sen1_toks, sen2_toks = map(
            word_tokenize,
            (line.decode('utf-8') for line in open(sen_file).readlines()))
        sen1_deps, sen2_deps = get_deps(file(dep_file))
        if not sen1_deps:
            return None, None
        elif not sen2_deps:
            return None, None

        whitelist_toks = set(sen1_toks + sen2_toks)
        graphs = (get_graph(sen1_deps, whitelist_toks, wrapper),
                  get_graph(sen2_deps, whitelist_toks, wrapper))

        if to_dot:
            for c, graph in enumerate(graphs):
                f = open('sen{}.dot'.format(c), 'w')
                f.write(graphs[c].to_dot())
                f.close()

        return graph_sim(graphs[0], graphs[1])

def main_batch(sens_path, deps_path, outs_path, cfg_file):
    print_now('loading wrapper...')
    wrapper = Wrapper(cfg_file, include_longman=True)
    print_now('reading sentence pairs...')

    assert os.path.isdir(sens_path) and os.path.isdir(deps_path), __USAGE__
    gold_path = os.path.join(os.environ['SEMEVAL_DATA'], 'sts_trial')
    assert os.path.isdir(gold_path), "{} does not exist".format(gold_path)
    for sen_fn in os.listdir(sens_path):
        sen_path = os.path.join(sens_path, sen_fn)
        dep_path = os.path.join(deps_path, sen_fn.replace('.sen', '.dep'))
        if not os.path.exists(dep_path):
            continue
        print_now('processing {}...'.format(sen_fn))
        sim, info = get_sim(sen_path, dep_path, wrapper)
        if sim is None:  # caused by empty dep files (parsing errors)
            continue
        gold_path = os.path.join(
            deps_path.replace('dep', 'gold'),
            sen_fn.replace('.sen', '.gold'))
        out_path = os.path.join(outs_path, sen_fn.replace('.sen', '.out'))
        gold_sim = float(file(gold_path).read().strip())
        open(out_path, 'w').write('sim: {}, gold: {}, {}\n'.format(
            sim, gold_sim, info))

def main_single(sens_path, deps_path, cfg_file):
    print_now('loading wrapper...')
    wrapper = Wrapper(cfg_file, include_longman=True)
    print_now('reading sentence pairs...')
    assert not os.path.isdir(deps_path), __USAGE__
    sim, info = get_sim(sens_path, deps_path, wrapper, to_dot=True)
    print_now('similarity: {}, info: {}'.format(sim, info))


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    if len(sys.argv) == 4:
        sens_path, deps_path, cfg_file = sys.argv[1:4]
        main_single(sens_path, deps_path, cfg_file)
    elif len(sys.argv) == 5:
        sens_path, deps_path, out_path, cfg_file = sys.argv[1:5]
        main_batch(sens_path, deps_path, out_path, cfg_file)
    else:
        raise Exception(__USAGE__)
