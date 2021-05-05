#!/usr/bin/env python
import os
import random
import tempfile
from collections import Counter

import matplotlib as mpl
import matplotlib.image as mpimg
import networkx as nx
import numpy as np
from IPython.display import Image
from matplotlib import rcParams
from networkx.drawing.nx_agraph import write_dot
from sklearn.cluster import DBSCAN

from qf.util import indexify


def addEdgesWithName(G, triples):
    """
        Adds to the graph G a set of arcs with names. Each element of triples is a triple (u,v,n) where u and v are the
        source and target of the arc and n is its name (it will be put in the "label" attribute of the edge).

        Args:
            G: the graph to which edges will be added (a `networkx.MultiDiGraph`).
            triples: a list of triples; each triple is a pair of nodes followed by a string (which will be stored in the "label" attribute of the edge).
                The same source/target pair may appear more than once, but it is advisable to have all labels distinct, and also labels distinct from
                node names.
    """
    lab = [name for u,v,name in triples]
    if len(lab) != len(set(lab)):
        print("Careful --- repeated edge name")
    for u,v,name in triples:
        G.add_edges_from([(u,v,{"label": name})])

# Removes from graph G the set of edges with given names ("label").
def removeEdgesWithNames(G, set_of_names):
    t = []
    for u,v,k,d in G.edges(data=True, keys=True):
        if d["label"] in set_of_names:
            t.append((u,v,k))
    for x in t:
        G.remove_edge(x[0], x[1], x[2])


def _visualize(GG, dot_filename, png_filename, colors=None, labelNodes=True, labelArcs=True):
    G = GG.copy()
    if colors != None:
        node_to_colindex=indexify(colors)
        colindex_to_rgb=list(mpl.colors.get_named_colors_mapping().values())
        for x in G.nodes:
            if x in node_to_colindex:
                G.nodes[x]["style"]="filled"
                G.nodes[x]["fillcolor"]=colindex_to_rgb[node_to_colindex[x]]
    if len(nx.get_node_attributes(G, "pos")) > 0:
        for s,t,d in G.edges(data=True):
            if "label" in d:
                del d["label"]
            d["arrowsize"]=0.3
        for x,d in G.nodes(data=True):
            d["fontsize"]=8
            d["height"]=0.3
            d["width"]=0.3            
            d["fixedsize"]=True
            d["esep"]=0
    if not labelArcs:
        for s,t,d in G.edges(data=True):
            if "label" in d:
                del d["label"]
    if not labelNodes:
        nx.set_node_attributes(G, {x: "" for x in G.nodes}, "label")       
    nx.nx_pydot.write_dot(G, dot_filename)
    if len(nx.get_node_attributes(G, "pos")) > 0:
        os.system("fdp -Goverlap=scale -Tpng -Gsplines=true {} -o {}".format(dot_filename, png_filename))
    else:
        os.system("dot -T png {} -o {}".format(dot_filename, png_filename))

# Visualizes the multidigraph G. If colors is not None, it must be a map with G.nodes as keys: if that happens,
# nodes will have the same color iff they have the same  value
def visualize(G, colors=None, labelNodes=True, labelArcs=True):
    dot_filename = tempfile.NamedTemporaryFile(suffix=".dot")
    png_filename = tempfile.NamedTemporaryFile(suffix=".png")
    _visualize(G, dot_filename, png_filename, colors, labelNodes, labelArcs)
    return Image(filename=png_filename)

# Same as visualize, but returns the pair of filenames where the dot and png file are saved. If they are
# not set, they are generated at random.
def save(G, dot_filename=None, png_filename=None, colors=None, labelNodes=True, labelArcs=True):
    if dot_filename is None:
        dot_filename = tempfile.NamedTemporaryFile(suffix=".dot")
    if png_filename is None:
        png_filename = tempfile.NamedTemporaryFile(suffix=".png")
    _visualize(G, dot_filename, png_filename, colors, labelNodes, labelArcs)
    return (dot_filename, png_filename)

# Creates a multidigraph that is fibred over G: fibres maps every node of G to the corresponding fibre.
# The source of every arc is chosen at random
def lift(G, fibre):
    H = nx.MultiDiGraph()
    for x in G.nodes:
        for v in fibre[x]:
            H.add_node((x,v))
    for source,target,d in G.edges(data=True):
        for v in fibre[target]:
            w=random.choice(fibre[source])
            addEdgesWithName(H, [((source, w), (target, v), d["label"] + "_" + str(v))])
    return H

# Given a graph and its Cardon-Crochemore labelling, returns the minimum base.
def minimum_base(G, node2class):
    B = nx.MultiDiGraph()
    class2node = {c:[x for x,klass in node2class.items() if klass==c][0] for c in node2class.values()}
    B.add_nodes_from(list(set(class2node.values())))
    for klass in B.nodes():
        for u,v,k in G.in_edges([klass], keys=True):
            B.add_edge(class2node[node2class[u]], klass, label="a_"+str(B.number_of_edges()))
    nx.set_node_attributes(B, {klass: str(klass)+" ("+str(len([x for x,v in node2class.items() if v==node2class[klass]]))+")" for klass in B.nodes()}, "label")
    nx.set_node_attributes(B, {klass: len([x for x,v in node2class.items() if v==node2class[klass]]) for klass in B.nodes()}, "multiplicity")
    return B


# Adds nAdd, deletes nDel, adds/deletes nScr edges from the graph G, returning the new graph
def scramble(G, nAdd=1, nDel=1, nScr=0):
    H = G.copy()
    for i in range(nAdd + nDel + nScr):
        operation = random.choice(["add", "delete"])
        if i < nAdd or i >= nAdd + nDel and operation == "add":
            e = random.choice([(u,v,d) for u,v,d in G.edges(data=True)])
            newname = e[2]["label"].split("_")[0] + "_" + str(random.randint(10000, 20000))
            addEdgesWithName(H, [(e[0],e[1],newname)])
        else:
            e = random.choice([e for e in H.edges])
            H.remove_edge(e[0],e[1])
    return H        



# Given a graph G and another graph B, builds and returns a pair of maps (for nodes and for arcs) as follows.
# Nodes of G are assumed to be of the form (x,y), and they should be mapped to node x of B.
# Arcs of G are assumed to be assigned a name ("name") of the form s_n, and they should be mapped to an arc ob B
# with name s
def extractLiftingQF(G, B):
    mn = {}
    ma = {}
    for x in G.nodes():
        mn[x] = x[0]
    for u, v, d in G.edges(data=True):
        ma[d["label"]] = d["label"].split("_")[0]
    return (mn, ma)

def to_simple(G):
    """
        Converts the graph G to a simple directed graph.

        Args:
            G (networkx.MultiDiGraph): a multidigraph.

        Returns:
            The simple directed (networkx.DiGraph) of `G`.
    """
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for x,y in G.edges():
        H.add_edge(x,y)
    return H

