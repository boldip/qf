"""
    All the functions in this module require that the input
    graphs have their arcs labelled (with an attribute called "label").
    Labels must be distinct and also disjoint from the names used
    for the nodes.
"""


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

import logging
import random
import statistics
import sys
from collections import Counter

import matplotlib as mpl
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import write_dot
from sklearn.cluster import DBSCAN

import qf.cc
import qf.graphs
import qf.qzss
import qf.util


def source(G, a):
    """
        Returns the source in G of the arc labelled a.

        Args:
            G: a `networkx.MultiDiGraph`.
            a: the label of an arc.

        Returns:
            the source of the arc.
    """
    t=[(x,y,d) for (x,y,d) in G.edges(data=True) if d["label"]==a]
    if len(t) != 1:
        return None
    return t[0][0]

# Returns the source in G of the arc labelled a.
def target(G, a):
    """
        Returns the target in G of the arc labelled a.

        Args:
            G: a `networkx.MultiDiGraph`.
            a: the label of an arc.

        Returns:
            the target of the arc.
    """
    t=[(x,y,d) for (x,y,d) in G.edges(data=True) if d["label"]==a]
    if len(t) != 1:
        return None
    return t[0][1]

def arcs(G):
    """
        Returns the set of labels of the arcs in G.

        Args:
            G: a `networkx.MultiDiGraph`.

        Returns:
            the set of labels of the arcs.
    """
    return set([d["label"] for (a,b,d) in G.edges(data=True)])

def get_arc(G, a):
    """
        Returns the tuple (x,y,key) of the arc of G with label a.

        Args:
            G: a `networkx.MultiDiGraph`.
            a: the label of an arc.

        Returns:
            the triple (x,y,key) where x is the arc source, y is its target 
            and key its key.
    """
    x,y,k,d = [(x,y,k,d) 
               for (x,y,k,d) in G.edges(data=True, keys=True) 
               if d["label"]==a][0]
    return x,y,k

# 
def new_arc_label(G):
    """
        Returns a new arc label for G (not yet present).

        Args:
            G: a `networkx.MultiDiGraph`.

        Returns:
            a label not belonging to the set of labels of the arcs of G.
    """
    Ga = arcs(G)
    result = "new_arc_" + str(random.randint(0, 1E9))
    while result in Ga:
        result = "new_arc_" + str(random.randint(0, 1E9))
    return result

def is_morphism(f, G, B):
    """
        Checks if f is a morphism between G and B; it must be a dictionary mapping 
        nodes to nodes and arcs (i.e. arc labels) to arcs (arc labels).

        Args:
            f (dict): a dictionary with keys the nodes and arc labels of G and values the nodes and arc labels of B.
            G: a `networkx.MultiDiGraph`.
            B: a `networkx.MultiDiGraph`.

        Returns:
            true iff the function is defined on all nodes, on all arcs, and the values are compatible as in the definition
            of a morphism. 
    """
    Ge = arcs(G)
    Be = arcs(B)
    # Check that maps nodes to nodes and arcs to arcs
    for k,v in f.items():
        if not(k in G.nodes) and not(k in Ge):
            logging.warning("Wrong key: {}".format(k))
            return False
        if not(v in B.nodes) and not(v in Be):
            logging.warning("Wrong value: {}".format(v))
            return False
        if (k in G.nodes) != (v in B.nodes):
            logging.warning("Type mismatch: {} vs. {}".format(k,v))
            return False
    # Check that all nodes and arcs are mapped
    for x in G.nodes():
        if not(x in f.keys()):
            logging.warning("Node {} not mapped".format(x))
            return False
    for a in Ge:
        if not(a in f.keys()):
            logging.warning("Arc {} not mapped".format(a))
            return False
    for a in f.keys():
        if a in Ge:
            if f[source(G,a)] != source(B,f[a]):
                logging.warning("Source of {}={}, source of {}={}".format(a,source(G,a),f[a],source(B,f[a])))
                return False
            if f[target(G,a)] != target(B,f[a]):
                logging.warning("Target of {}={}, target of {}={}".format(a,target(G,a),f[a],target(B,f[a])))
                return False
    return True

def is_epimorphism(f, G, B):
    """
        Check if f is an epimorphism from G to B. Among other things, it also
        checks if f is a morphism.

        Args:
            f (dict): a dictionary with keys the nodes and arc labels of G and values the nodes and arc labels of B.
            G: a `networkx.MultiDiGraph`.
            B: a `networkx.MultiDiGraph`.

        Returns:
            true iff the function is a morphism, and the value set contains all nodes and all arcs of B. 
    """
    if not is_morphism(f, G, B):
        return False
    if set(B.nodes) != set(f[x] for x in G.nodes):
        logging.warning("Not epimorphic on nodes")
        return False
    if arcs(B) != set(f[a] for a in arcs(G)):
        logging.warning("Not epimorphic on arcs")
        return False
    return True

def is_isomorphism(f, G, B):
    """
        Check if f is an isomorphism from G to B. Among other things, it also
        checks if f is a morphism.

        Args:
            f (dict): a dictionary with keys the nodes and arc labels of G and values the nodes and arc labels of B.
            G: a `networkx.MultiDiGraph`.
            B: a `networkx.MultiDiGraph`.

        Returns:
            true iff the function is an isomorphism.
    """
    if not is_epimorphism(f, G, B):
        return  False
    # Injective on nodes
    for x in G.nodes():
        for y in G.nodes():
            if x != y and f[x] == f[y]:
                return False
    # Injective on arcs
    for a in qf.morph.arcs(G):
        for b in qf.morph.arcs(G):
            if a != b and f[a] == f[b]:
                return False
    return True

def is_compatible(G, Gp, f=None, fp=None):
    """
        Check if G and Gp are compatible (i.e., same node set, compatible source/target for common arcs);
        unless they are None, the function also checks that f and fp have the same value on common arcs and nodes.

        Args:
            G: a `networkx.MultiDiGraph`.
            Gp: a `networkx.MultiDiGraph`.
            f (dict): a dictionary representing a morphism from G to some other graph (see `is_morphism`), or None.
            fp (dict): a dictionary representing a morphism from Gp to some other graph (see `is_morphism`), or None.

        Returns:
            true iff G and Gp are compatible, and same for f and fp.
    """
    if (f is None) != (fp is None):
        raise Exception("Either both f and fp are None, or neither") 
    if G.nodes() != Gp.nodes():
        return False
    if f is not None:
        for x in G.nodes():
            if f[x] != fp[x]:
                return False
    for a in qf.morph.arcs(G) & qf.morph.arcs(Gp):
        if qf.morph.source(G, a) != qf.morph.source(Gp, a):
            return False
        if qf.morph.target(G, a) != qf.morph.target(Gp, a):
            return False
        if f is not None and f[a] != fp[a]:
            return False
    return True


def excess_deficiency(f, G, B, verbose=False):
    """
        Given a morphism f from G to B, computes and returns its excess and deficiency. These two measures are defined
        locally for any given pair (a,x) where a is an arc of B and x is a node of G with f(x) being the target of a:
        for the specific pair (a,x), count the number of arcs a' of G such that f(a')=a and the target of a' is x.
        If this count is larger than one, the difference with one is called the *local excess*. If the count is zero,
        we say that one is the *local deficiency*. The sum of all local excesses and local deficiencies over all the
        pairs (a,x) is called the excess and deficiency. By definition, excess and deficiency is zero iff f is a fibration.

        Args:
            f (dict): a morphism from G to B (see `is_morphism`).
            G: a `networkx.MultiDiGraph`.
            B: a `networkx.MultiDiGraph`.
            verbose (bool): if True, local non-null excesses and deficiencies are printed out.

        Returns:
            (x,d) (a pair of int), where x is the excess and d is the deficiency.
    """
    deficiency = 0
    excess = 0
    for a in arcs(B):
        ta = target(B, a)
        targetsa = set([x for x in G.nodes if f[x]==ta])
        for x in targetsa:
            ts = set([ap for ap in arcs(G) if f[ap]==a and target(G, ap)==x])
            if len(ts) == 0:
                if verbose:
                    print("DEFICIENCY: cannot lift arc {} at {}".format(a, x))
                deficiency += 1
            else:
                if verbose and len(ts)>1:
                    print("EXCESS: lifting arc {} at {} gives {} excess results".format(a, x, len(ts) - 1))
                excess += len(ts) - 1
    return (excess, deficiency)

def is_fibration(f, G, B):
    """
        Determines if a dictionary is a fibration. It also checks if it is a morphism.

        Args:
            f (dict): a dictionary representing a morphism (see `is_morphism`).
            G: a `networkx.MultiDiGraph`.
            B: a `networkx.MultiDiGraph`.

        Returns:
            True iff f is a morphism and a fibration.
    """
    if not is_morphism(f, G, B):
        return False
    excess, deficiency = excess_deficiency(f, G, B)
    return excess+deficiency == 0

def repair(f, G, B, seed=0, verbose=False):
    """
        This is the implementation of GraphRepair. Given a morphism f: G->B with excess x and deficiency d,
        it will build a new compatible graph G' (i.e., with the same nodes as G and such that the common arcs, if any, have the same source
        and target as in G) and a new morphism f': G'->B such that: 

        - the symmetric difference between the arcs of G and the arcs of G' has cardinality x+d
        - f is defined in the same way as f' on all nodes and on common arcs
        - f' is a fibration.

        The construction of G' and f' are partly non-deterministic. For this reason, randomness is used.

        Args:
            f (dict): a morphism (see `is_morphism`).
            G: a `networkx.MultiDiGraph`.
            B: a `networkx.MultiDiGraph`.
            seed (int): the seed used for the non-deterministic part. If None, a new random seed is set.
            verbose (bool): if True, removed or added arcs are printed out.

        Returns:
            a pair (Gp, fp) where Gp is a `networkx.MultiDiGraph` and fp is a dict, as described above.

    """
    if seed is None:
        seed = random.randrange(sys.maxsize)
        logging.info("Seed for repair set to {}".format(seed))
    random.seed(seed)
    Gp = G.copy()
    fp = f.copy()
    for a in arcs(B):
        ta = target(B, a)
        targetsa = set([x for x in G.nodes if f[x]==ta])
        for x in targetsa:
            ts = [ap for ap in arcs(G) if f[ap]==a and target(G, ap)==x]
            if len(ts) == 1:
                continue
            if len(ts) == 0:
                a_to_add = new_arc_label(Gp)
                src = random.sample([s for s in G.nodes if f[s]==source(B,a)],1)[0]
                qf.graphs.add_edges_with_name(Gp,[(src, x, a_to_add)])
                if verbose:
                    print("Adding arc {}: {} -> {} (mapped to {})".format(a_to_add, src, x, a))
                fp[a_to_add] = a
            else:
                ts = list(set(ts).difference(random.sample(ts,1)))
                for a_to_remove in ts:
                    xr,yr,kr = get_arc(Gp, a_to_remove)
                    if verbose:
                        print("Removing arc {}: {} -> {}".format(a_to_remove, xr, yr))
                    Gp.remove_edge(xr,yr,key=kr)
                    del fp[a_to_remove]
    return (Gp, fp)

def qf_build(G, c, verbose=False):
    """
        This is the implementation of QFBuild. Given an equivalence relation c on the nodes of G (represented as a dict: the keys are nodes and two
        nodes are equivalent iff they have the same value), build a graph B and a morphism f: G->B with the following properties:
        
        - the node fibres of f (i.e., counterimages of nodes) are the equivalence classes of c
        - there is no other f': G->B' with smaller excess+deficiency.

        Args:
            G: a `networkx.MultiDiGraph`.
            c (dict): an equivalence relation on the nodes of G; the keys are nodes, and two nodes are equivalent iff they have the same values
            verbose (bool): if True, for any pair of equivalence classes X and Y, if there is at least one arc from a node of X to a node of Y,
                a line is printed with X, Y and the list of the 
                number of arcs from (any node of) X to every specific node of Y; the resulting median is also printed.

        Returns:
            a pair (B, f) where B is a `networkx.MultiDiGraph` and f is a dict, as described above.

    """
    B = nx.MultiDiGraph()
    f = {}
    classes = set(c.values())
    for klass in classes:
        B.add_node(klass)
    for node in G.nodes():
        f[node] = c[node]
    for target_klass in classes:
        for source_klass in classes:
            v = []
            for y in [x for x in G.nodes if c[x]==target_klass]:
                count = 0
                for s,t,a in G.edges(data=True):
                    if t == y and c[s] == source_klass:
                        count += 1
                v.append(count)
            if sum(v) == 0:
                continue
            k = max(int(statistics.median(v)), 1) #WHY MAX
            if verbose:
                print("{} -> {}: {} median {}".format(source_klass, target_klass, v, k))
            arc_label = []
            for i in range(k):
                arc_label.append("{}_{}_{}".format(source_klass, target_klass, i))
                qf.graphs.add_edges_with_name(B, [
                    (source_klass, target_klass, 
                     arc_label[i]
                    )
                ])
            for y in [x for x in G.nodes if c[x]==target_klass]:
                count = 0
                for s,t,a in G.edges(data=True):
                    if t == y and c[s] == source_klass:
                        f[a["label"]] = arc_label[count % len(arc_label)]
                        count += 1
    return (B, f)


