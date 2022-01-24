#!/usr/bin/env python

#
# qf: Quasi-Fibrations of Graphs
#
# Copyright (C) 2021-2026 Paolo Boldi
#
# This program and the accompanying materials are made available under the
# terms of the GNU Lesser General Public License v2.1 or later,
# which is available at
# http://www.gnu.org/licenses/old-licenses/lgpl-2.1-standalone.html,
# or the Apache Software License 2.0, which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.
#
# SPDX-License-Identifier: LGPL-2.1-or-later OR Apache-2.0
#


import random
from collections import Counter

import matplotlib as mpl
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import write_dot
from sklearn.cluster import DBSCAN

from qf.util import indexify


def cardon_crochemore(G, max_step=-1):
    """
        Computes and returns the coarsest equitable partition of the nodes of the graph G.
        If `max_step` is not negative, the computation is stopped prematurely after that number of iterations.

        Args:
            G: a `networkx.MultiDiGraph`.
            max_step (int): the maximum number of iterations, or -1 to stop only after `G.number_of_nodes()` steps.

        Returns:
            a dictionary with the nodes of `G` as keys, and with values in `range(k)` where `k` is the number of
            classes; two nodes have the same value iff they belong to the same class.
    """
    if max_step < 0:
        max_step = len(G.nodes)
    label={x:0 for x in G.nodes}
    n=len(set(label.values()))
    i = 0
    while True:
        m={}
        for v in G.nodes:
            t=[]
            for w in G.in_edges([v]):
                t.append(label[w[0]])
            t.sort()
            t.append(label[v])
            m[v]=t
        label=indexify(m)
        nn=len(set(label.values()))
        if nn==n:
            break
        n=nn
        i += 1
        if i >= max_step:
            break
    return label



