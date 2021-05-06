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

def removeEdgesWithNames(G, set_of_names):
    """
        Removes grom a given graph the edges with a given "label" attribute.

        Args:
            G: the graph to modify (a `networkx.MultiDiGraph`; all of its edges must have a "label" attribute).
            set_of_names (list): the set of labels of the edges to be removed.
    """
    t = []
    for u,v,k,d in G.edges(data=True, keys=True):
        if d["label"] in set_of_names:
            t.append((u,v,k))
    for x in t:
        G.remove_edge(x[0], x[1], x[2])


def _visualize(GG, dot_filename, png_filename, colors=None, labelNodes=True, labelArcs=True):
    """
        It writes out a graph in dot and png format, on two given files.
        - If `colors` are provided, they must be a dict mapping nodes to values (distinc values for distinc colors).
        In this case, alle nodes will be given a "style"="filled" attribute and a "fillcolor" equal to the name of a color.
        - If nodes has a `pos` attribute, the following edge and node attributes
        are fixed: edges -> "arrowsize"=0.3; nodes -> "fontsize"=8, "height"=0.3, "width"=0.3, "fixedsize"=True, "esep"=0. 
        - If nodes has a `pos` attribute, or `labelArcs` is False, all "label" edge attributes are removed.
        - If `labelNodes` is False, all "label" node attributes are set to "".
        - The rendering for png is performed using `dot` or `fdp` depending on whether nodes have a `pos` attribute or not. 

        Attrs:
            GG: a `nerworkx.MultiDiGraph`; edges should all possess a `label` attribute.
            dot_filename (str): the name of the file where the .dot output will  be written.
            png_filename (str): the name of the file where the .png output will be written.
            colors (dict): if not None, a dictionary from nodes to values (same value mean same color).
            labelNodes (bool): whether labels should appear on nodes (the actual node name).
            labelArcs (bool): whether labels should appear on arcs (the "label" attribute on edges).
    """
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

def visualize(G, colors=None, labelNodes=True, labelArcs=True):
    """
        Returns an image for the given graph. 
        - If `colors` are provided, they must be a dict mapping nodes to values (distinc values for distinc colors).
        In this case, alle nodes will be given a "style"="filled" attribute and a "fillcolor" equal to the name of a color.
        - If nodes has a `pos` attribute, the following edge and node attributes
        are fixed: edges -> "arrowsize"=0.3; nodes -> "fontsize"=8, "height"=0.3, "width"=0.3, "fixedsize"=True, "esep"=0. 
        - If nodes has a `pos` attribute, or `labelArcs` is False, all "label" edge attributes are removed.
        - If `labelNodes` is False, all "label" node attributes are set to "".
        - The rendering for png is performed using `dot` or `fdp` depending on whether nodes have a `pos` attribute or not. 

        Attrs:
            G: a `nerworkx.MultiDiGraph`; edges should all possess a `label` attribute.
            colors (dict): if not None, a dictionary from nodes to values (same value mean same color).
            labelNodes (bool): whether labels should appear on nodes (the actual node name).
            labelArcs (bool): whether labels should appear on arcs (the "label" attribute on edges).
    """
    dot_filename = tempfile.NamedTemporaryFile(suffix=".dot")
    png_filename = tempfile.NamedTemporaryFile(suffix=".png")
    _visualize(G, dot_filename, png_filename, colors, labelNodes, labelArcs)
    return Image(filename=png_filename)

def save(G, dot_filename=None, png_filename=None, colors=None, labelNodes=True, labelArcs=True):
    """
        Writes a given graph in dot and png formats. If names of files are not provided, they are generated by
        this function.
        - If `colors` are provided, they must be a dict mapping nodes to values (distinc values for distinc colors).
        In this case, alle nodes will be given a "style"="filled" attribute and a "fillcolor" equal to the name of a color.
        - If nodes has a `pos` attribute, the following edge and node attributes
        are fixed: edges -> "arrowsize"=0.3; nodes -> "fontsize"=8, "height"=0.3, "width"=0.3, "fixedsize"=True, "esep"=0. 
        - If nodes has a `pos` attribute, or `labelArcs` is False, all "label" edge attributes are removed.
        - If `labelNodes` is False, all "label" node attributes are set to "".
        - The rendering for png is performed using `dot` or `fdp` depending on whether nodes have a `pos` attribute or not. 

        Attrs:
            G: a `nerworkx.MultiDiGraph`; edges should all possess a `label` attribute.
            dot_filename (str): the name of the file where the .dot output will  be written; if None, the name will be generated.
            png_filename (str): the name of the file where the .png output will be written; if None, the name will be generated.
            colors (dict): if not None, a dictionary from nodes to values (same value mean same color).
            labelNodes (bool): whether labels should appear on nodes (the actual node name).
            labelArcs (bool): whether labels should appear on arcs (the "label" attribute on edges).

        Returns:
            the pair (dot_filename, png_filename) (if either is None, it is generated by this function).
    """
    if dot_filename is None:
        dot_filename = tempfile.NamedTemporaryFile(suffix=".dot")
    if png_filename is None:
        png_filename = tempfile.NamedTemporaryFile(suffix=".png")
    _visualize(G, dot_filename, png_filename, colors, labelNodes, labelArcs)
    return (dot_filename, png_filename)

def lift(G, fibre):
    """
        Given a graph G and a dictionary `fibre` from the nodes of G to a list, it produces a new graph
        where nodes are pairs (y,v) (with v in `fibre[y]`) and for every arc (x,y) there is
        an arc incoming in each (y,v), whose source is is (x,w) for a randomly chosen w in `fibre[x]`.
        The label associated to the arc is the label of the original arc followed by "_" and v.

        
        Args:
            G: a `networkx.MultiDiGraph`, whose arcs have a "label" attribute.
            fibre (dict): a map from nodes of G to lists.

        Returns:
            a graph as described above: this graph is fibred on G with fibres corresponding to the
            values of `fibre`.
    """
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

