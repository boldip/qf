
/*
 * qf: Quasi-Fibrations of Graphs
 *
 * Copyright (C) 2021-2026 Paolo Boldi
 *
 * This program and the accompanying materials are made available under the
 * terms of the GNU Lesser General Public License v2.1 or later,
 * which is available at
 * http://www.gnu.org/licenses/old-licenses/lgpl-2.1-standalone.html,
 * or the Apache Software License 2.0, which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later OR Apache-2.0
 */

import random
from collections import Counter

import matplotlib as mpl
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import write_dot

import qf.cc
import qf.graphs
import qf.qzss
import qf.util


# Generate a graph with n nodes, get its minimum base and then
# lift every node with a fibre of [vmin..vmax] elements.
# Nodes have the form (x,y) where x denotes the fibre.
def get_fibration_rich_graph(n = 10, vmin = 1, vmax = 5, verbose=True):
    """
        Returns a fibration-rich graph (a graph with some nontrivial fibres w.r.t. minimal fibrations).
        The graph is generated as follows: first a `networkx.scale_free_graph` with n nodes is produced,
        and then its minimal base B is obtained.
        Further every node x of B is replicated a random number of times in the range [vmin..vmax].
        The resulting replicas have all the same indegree as x, and are rewired to suitable replicas of their
        in-neighbours. 

        Args:
            n: the number of nodes of the scale free graph we start from.
            vmin: the minimum number of nodes in a fibre.
            vmax: the maximum number of nodes in a fibre.
            verbose: whether the function should produce output during the computation.

        Returns:
            the resulting graph.
    """
    GG =nx.MultiDiGraph(nx.scale_free_graph(n))                             # Generate the graph
    if verbose:
        print("Original graph: n={}, nodes={}, arcs={}".format(n, GG.number_of_nodes(), GG.number_of_edges()))
    B = qf.graphs.minimum_base(GG, qf.cc.cardon_crochemore(GG))             # Minimum base
    if verbose:
        print("Minimum base: nodes={}, arcs={}".format(B.number_of_nodes(), B.number_of_edges()))
    G=qf.graphs.lift(B, {k: range(random.randint(vmin,vmax)) for k in B.nodes}) 
    if verbose:
        print("Result: nodes={}, arcs={}".format(G.number_of_nodes(), G.number_of_edges()))
    return G

