#!/usr/bin/env python

import logging
import sys

from pymachine.src.machine import MachineGraph
from pymachine.src.wrapper import Wrapper

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

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    print 'building wrapper...'

    sen1_deps, sen2_deps = get_deps(sys.stdin)

    for c, dep_list in enumerate((sen1_deps, sen2_deps)):
        w = Wrapper(sys.argv[1])
        for dep in dep_list:
            w.add_dependency(dep)
        active_machines = w.lexicon.active_machines()
        graph = MachineGraph.create_from_machines(active_machines)
        f = open('sen{}.dot'.format(c), 'w')
        f.write(graph.to_dot())
        f.close()

if __name__ == '__main__':
    main()
