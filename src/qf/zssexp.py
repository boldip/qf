#!/usr/bin/env python


import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib as mpl
import random
from collections import Counter
import numpy as np
import qf.cc
import qf.graphs
import qf.util
import qf.qzss

# Generate a graph with n nodes, get its minimum base and then
# lift every node with a fibre of [vmin..vmax] elements.
# Nodes have the form (x,y) where x denotes the fibre.
def getFibrationRichGraph(n = 10, vmin = 1, vmax = 5, verbose=True):
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

# Returns a new graph, obtained copying G and removing rem arcs.
def removeEdges(G, rem = 1):
    Gs = G.copy()
    removed = random.sample(list(G.edges), rem)
    Gs.remove_edges_from(removed)
    return Gs


# A number expno of experiments is run. Every experiment consists in the following: 
# - a fibration rich graph L is generated (calling getFibrationRichGraph(n, vmin, vmax)
# - the map from the nodes of this graph and the fiber they belong to is called the "ground truth"
# - a random set of edges, between remmin (inclusive) and remmax (exclusive), are removed; the resulting graph is called Ls
# - Cardon-Crochemore is run on Ls
# - agglomerative clustering is run on Ls using the same number of clusters as in the ground truth
# - the function prints (in this order):
#       - experiment number
#       - number of nodes of Ls
#       - number of edges Ls
#       - number of edges that were removed
#       - adjusted rand index between Cardon-Crochemore and the ground truth
#       - adjusted rand index between agglomerative clustering at depth depth and the ground truth

def bigExperiment(expno = float("inf"), depth = 2, n = 10, vmin = 1, vmax = 5, remmin = 1, remmax = 4):
    x=[]
    y=[]
    i = 0
    while (i < expno):
        L = getFibrationRichGraph(n, vmin, vmax)
        rem = random.randint(remmin ,remmax)
        Ls = removeEdges(L, rem)
        groundTruthLab = {x: x[0] for x in L.nodes}
        groundTruthSize = len(set(groundTruthLab.values()))
        if groundTruthSize == 0:
            continue
        ccLab = qf.cc.cardon_crochemore(Ls)                                      
        clustering, M, nodes, indices = qf.qzss.agclust(Ls, depth, groundTruthSize)
        agLab = qf.qzss.agclust2dict(clustering, M, nodes, indices)
        print(i, len(Ls.nodes), len(Ls.edges), rem, qf.util.nmi(groundTruthLab, ccLab), qf.util.nmi(groundTruthLab, agLab), flush=True)
        i += 1

# A number expno of experiments is run. Every experiment consists in the following: 
# - a fibration rich graph L is generated (calling getFibrationRichGraph(n, vmin, vmax)
# - the map from the nodes of this graph and the fiber they belong to is called the "ground truth"
# - a random set of edges, between remmin (inclusive) and remmax (exclusive), are removed; the resulting graph is called Ls
# - Cardon-Crochemore is run on Ls
# - agglomerative clustering is run on Ls using an optimal number of clusters (as in agclustOpt)
# - the function prints (in this order):
#       - experiment number
#       - number of nodes of Ls
#       - number of edges Ls
#       - number of edges that were removed
#       - adjusted rand index between Cardon-Crochemore and the ground truth
#       - adjusted rand index between agglomerative clustering at depth depth and the ground truth

def bigExperimentCl(expno = float("inf"), depth = 2, n = 10, vmin = 1, vmax = 5, remmin = 1, remmax = 4):
    x=[]
    y=[]
    i = 0
    while (i < expno):
        L = getFibrationRichGraph(n, vmin, vmax)
        rem = random.randint(remmin ,remmax)
        Ls = removeEdges(L, rem)
        groundTruthLab = {x: x[0] for x in L.nodes}
        groundTruthSize = len(set(groundTruthLab.values()))
        if groundTruthSize == 0:
            continue
        ccLab = qf.cc.cardon_crochemore(Ls)                                      
        clustering, M, nodes, indices = qf.qzss.agclustOptcl(Ls, depth)
        agLab = qf.qzss.agclust2dict(clustering, M, nodes, indices)
        print(i, len(Ls.nodes), len(Ls.edges), rem, qf.util.nmi(groundTruthLab, ccLab), qf.util.nmi(groundTruthLab, agLab), flush=True)
        i += 1

