#!/usr/bin/env python
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib as mpl
import random
from collections import Counter
from sklearn.cluster import DBSCAN
import numpy as np

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


# Given two vectors a,b compute their Jaccard similarity: for every element x appearing in either a or b (or both), let m(x) and M(x) be the minimum and maximum of the number
# of occurrences of x in a and b. The Jaccard similarity is the sum of m(x) divided by the sum of M(x).
# If a and b contain no repetitions, the value returned is the Jaccard similarity of set(a) and set(b).
# The returned value is a similarity value between 0 and 1 (1=same multisets).
def jaccard_similarity(a, b):
    mins = 0
    maxs = 0
    for x in set(a) | set(b):
        xa = a.count(x)
        xb = b.count(x)
        mins += min(xa, xb)
        maxs += max(xa, xb)
    if maxs == 0:
        return 1
    return mins / maxs

# Returns the soft Cardon-Crochemore labelling (a map from nodes whose fibres are the same as in the CC algorithm)
def soft_cardon_crochemore(G, epsilon=0.1):
    label={x:0 for x in G.nodes}
    n=len(set(label.values()))
    for i in range(len(list(G.nodes))):
        db=DBSCAN(metric="precomputed", eps=epsilon)
        m={}
        for v in G.nodes:
            t=[]
            for w in G.in_edges([v]):
                t.append(label[w[0]])
            m[v]=t
        nodes = list(G.nodes)
        nnodes = len(nodes)
        dd = []
        for i in range(nnodes):
            d = []
            for j in range(nnodes):
                d.append(1-jaccard_similarity(m[nodes[i]], m[nodes[j]]))
            dd.append(d)
        db.fit(np.array(dd))
        label=indexify({nodes[i]: [db.labels_[i],label[nodes[i]]] for i in range(nnodes)})
        nn=len(set(label.values()))
        if nn==n:
            break
        n=nn
    return label


