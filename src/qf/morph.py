"""
    All the functions in this module require that the input
    graphs have their arcs labelled (with an attribute called "label").
    Labels must be distinct and also disjoint from the names used
    for the nodes.
"""


import random
import statistics
from collections import Counter

import matplotlib as mpl
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import write_dot
from sklearn.cluster import DBSCAN

import qf.cc
import qf.graphs
import qf.util
import qf.zss
import qf.zssexp



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
            print("Wrong key: {}".format(k))
            return False
        if not(v in B.nodes) and not(v in Be):
            print("Wrong value: {}".format(v))
            return False
        if (k in G.nodes) != (v in B.nodes):
            print("Type mismatch: {} vs. {}".format(k,v))
            return False
    # Check that all nodes and arcs are mapped
    for x in G.nodes():
        if not(x in f.keys()):
            print("Node {} not mapped".format(x))
            return False
    for a in Ge:
        if not(a in f.keys()):
            print("Arc {} not mapped".format(a))
            return False
    for a in f.keys():
        if a in Ge:
            if f[source(G,a)] != source(B,f[a]):
                print("Source of {}={}, source of {}={}".format(a,source(G,a),f[a],source(B,f[a])))
                return False
            if f[target(G,a)] != target(B,f[a]):
                print("Target of {}={}, target of {}={}".format(a,target(G,a),f[a],target(B,f[a])))
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
        print("Not epimorphic on nodes")
        return False
    if arcs(B) != set(f[a] for a in arcs(G)):
        print("Not epimorphic on arcs")
        return False
    return True

# Computes the excess and deficiency of f: G to B.
def excess_deficiency(f, G, B, verbose=False):
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

# Determines if f: G to B is a fibration
def is_fibration(f, G, B):
    if not is_morphism(f, G, B):
        return False
    excess, deficiency = excess_deficiency(f, G, B)
    return excess+deficiency == 0

# Repairs a quasifibration f: G->B to a fibration f": G"->B.
# If seed=None a new random seed is set.
def repair(f, G, B, seed=0, verbose=False):
    random.seed(seed)
    Gp = G.copy()
    fp = f.copy()
    for a in arcs(B):
        ta = target(B, a)
        targetsa = set([x for x in G.nodes if f[x]==ta])
        for x in targetsa:
            ts = set([ap for ap in arcs(G) if f[ap]==a and target(G, ap)==x])
            if len(ts) == 1:
                continue
            if len(ts) == 0:
                a_to_add = new_arc_label(Gp)
                src = random.sample(set([s for s in G.nodes if f[s]==source(B,a)]),1)[0]
                qf.graphs.addEdgesWithName(Gp,[(src, x, a_to_add)])
                if verbose:
                    print("Adding arc {}: {} -> {} (mapped to {})".format(a_to_add, src, x, a))
                fp[a_to_add] = a
            else:
                ts = ts.difference(random.sample(ts,1))
                for a_to_remove in ts:
                    xr,yr,kr = get_arc(Gp, a_to_remove)
                    if verbose:
                        print("Removing arc {}: {} -> {}".format(a_to_remove, xr, yr))
                    Gp.remove_edge(xr,yr,key=kr)
                    del fp[a_to_remove]
    return (Gp, fp)

# Given a graph G and a labelling c (i.e., an equivalence relation on
# its nodes), build a graph B and a quasi-fibration f: G -> B with
# minimal total error
def qf_build(G, c, verbose=False):
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
                qf.graphs.addEdgesWithName(B, [
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


