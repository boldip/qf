import random
from collections import Counter

import matplotlib as mpl
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import write_dot
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score


def indexify(m):
    """
        Given a map `m`, make it into another map
        with the same keys and integer values, with the property that two keys are mapped to the same
        value iff they were mapped to the same value by `m` (more precisely, to two values with the same `str`).

        Args:
            m (dict): the input dictionary.

        Returns:
            the resulting dictionary.
    """
    values=list(set([str(v) for v in m.values()]))
    return {k: values.index(str(m[k])) for k in m.keys()}



def jaccard_multiset(x, y):
    """
        Given two lists (seen as multisets) returns their Jaccard coefficient (d(x,y)=sum_i min(x_i,y_i)/max(x_i,y_i) where
        x_i and y_i is the cardinality of i in `x` and `y` (resp.)).

        Args:
            x, y (list): a list seen as a multiset (the elements of the multiset are the elements of the list, the
                number of occurrences of each element in the list is seen as the number of replicas of the element
                in the corresponding multiset).

        Returns:
            the Jaccard coefficient (1 if both sets are empty).
        
    """
    cx = Counter(x)
    cy = Counter(y)
    num = 0
    den = 0
    for i in cx.keys() | cy.keys():
        num += min(cx[i], cy[i])  
        den += max(cx[i], cy[i])
    if den == 0:
        return 1
    return num / den


def readGraph(filename, skipHeader=True, separator="\t", dense=False, coordinates=None, scale=10):
    """
        Reads a graph from the given file, possibly skipping the first line (a.k.a. header). If not dense, every line is a `separator`-separated pair of node
        names, each corresponding to an arc. If dense, the header lines contains a first special value (ignored), the separator, and then `separator`-separated target names:
        the following lines contain each the source name, the separator, and then values (0 if there is no arc to the corresponding target, a non-zero value gives the
        number of arcs to the target).
        The result is a `networkx.MultiDiGraph`, whose nodes are strings (the node names in the file) and whose arcs have a "label" attribute with
        value "(s,t)" where s is the source and t is the target.
        If `coordinates` is provided, then every node has a "pos" attribute, with value "x,y!" where the coordinates are those associated
        to the node, each multiplied by `scale`.

        Args:
            filename (str): the name of the file containing the graph.
            skipHeader (bool): whether the first line should be skipped (ignored if dense==True).
            separator (str): the separator.
            dense (bool): whether the format is dense or not (see above).
            coordinates (dict): either `None`, or a dict whose keys are the nodes and whose values are tuples of two numbers each, representing the coordinates.
            scale (float): each coordinate is multiplied by this factor (see above).

        Returns:
            a `networkx.MultiDiGraph` as described above.
    """
    f = open(filename, "r")
    G = nx.MultiDiGraph()
    if skipHeader:
        t = f.readline()
        if dense:
            targets = t.strip().split(separator)[1:]
    while True:
        line = f.readline()
        if not line:
            break
        if dense:
            v = line.strip().split(separator)
            source = v[0]
            for i in range(1, len(v)):
                if int(v[i]) > 0 and targets[i - 1] != source:
                    for times in range(int(v[i])):
                        G.add_edge(source, targets[i - 1], label="(%s -> %s)" % (source, targets[i - 1]))
        else:
            if line.strip() == '':
                continue
            v = line.strip().split(separator)
            if v[0] != v[1]:
                G.add_edge(v[0], v[1], label="(%s -> %s)" % (v[0], v[1]))
    f.close()
    if coordinates is not None:
        t={}
        for k,v in coordinates.items():
            t[k]="{},{}!".format(v[0] * scale, v[1] * scale)        
        nx.set_node_attributes(G, t, "pos")
    return G


def readLabel(filename, skipHeader=True, separator="\t"):
    """
        Reads a label set from the given file, possibly skipping the first line. Every line is a `separator`-separated pair of node
        name and label,

        Args:
            filename (str): the name of the file containing the labels.
            skipHeader (bool): whether the first line should be skipped.
            separator (str): the separator between node name and label.

        Returns:
            a dictionary whose keys are the nodes (first column of the file) and whose values are the keys (second column).
    """
    f = open(filename, "r")
    m = {}
    if skipHeader:
        f.readline()
    while True:
        line = f.readline()
        if not line.strip():
            break
        v = line.strip().split(separator)
        m[v[0]] = v[1]
    f.close()
    return m

def readCoordinates(filename, skipHeader=True, separator=" "):
    """
        Reads the coordinates of nodes from a given file, possibly skipping the first line. Every line is a separator-separated triple:
        the first element is the node name, the remaining two elements are the X and Y coordinates (two floats), respectively.

        Args:
            filename (str): the name of the file containing the coordinatess.
            skipHeader (bool): whether the first line should be skipped.
            separator (str): the separator between node name, the X coordinate and the Y coordinate.

        Returns:
            a dictionary whose keys are the nodes (first column of the file) and whose values are pairs of floats (second and third column).

    """
    f = open(filename, "r")
    xy = {}
    if skipHeader:
        f.readline()
    while True:
        line = f.readline()
        if not line.strip():
            break
        v = line.strip().split(separator)
        xy[v[0]] = (float(v[1]),float(v[2]))
    f.close()
    return xy

# 
# Only common keys are returned
def nmi(lab1, lab2):
    """
        Given two clusterings (in the form of maps from nodes to clusters) returns their NMI.

        Args:
            lab1, lab2 (dict): maps from nodes to values (representing cluster identifiers).
    
        Returns:
            the normalized mutual information score adjusted for chance (`sklearn.metrics.adjusted_mutual_info_score`).
    """
    s = list(set(lab1.keys()) & set(lab2.keys()))
    l1 = []
    l2 = []
    for x in s:
        l1.append(lab1[x])
        l2.append(lab2[x])
    return adjusted_mutual_info_score(l1, l2)
    

def colors(n): 
    """
        Returns a list of random colors of given length.

        Args:
            n (int): the number of colors to be generated.

        Returns:
            a list with `n` colors, each represented as an RGB [0,1] triple.
    """
    ret = [] 
    r = int(random.random() * 256) 
    g = int(random.random() * 256) 
    b = int(random.random() * 256) 
    step = 256 / n 
    for i in range(n): 
        r += step 
        g += step 
        b += step 
        r = int(r) % 256 
        g = int(g) % 256 
        b = int(b) % 256 
        ret.append((r/256,g/256,b/256))  
    return ret 
