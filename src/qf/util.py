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
import signal
from collections import Counter
from contextlib import contextmanager
from queue import Queue

import logging
import matplotlib as mpl
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import write_dot
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score

import qf.uted.uted


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


def read_graph(filename, skipHeader=True, separator="\t", dense=False, coordinates=None, scale=10):
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


def read_label(filename, skipHeader=True, separator="\t"):
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

def read_coordinates(filename, skipHeader=True, separator=" "):
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

def dfs_tree(G, x, depth, i=0):
    """
        Consider the truncated universal total graph of x in G, and number its nodes in DFS (i is the root, i+1 is root's leftmost child, i+2 is its root's leftmost grandchild and so on).
        Let k be the number of nodes of the tree. 
        This function returns two lists:
            
        - the first is a list with the k nodes visited;
        - the second is a list of k lists, where the j-th list is the list of children of node j, in left-to-right order.

        Args:
            G: a `networkx.MultiDiGraph`.
            x: a node of G.
            depth: the truncation depth.
            i: the number of the first visited node
    """
    if depth == 0:
        return ([x], [[]]) 
    nodes = [x]
    adj = []
    children = []
    i += 1
    for s,t,d in G.in_edges(x, data=True):
        children.append(i)
        n, a = dfs_tree(G, s, depth - 1, i)
        nodes.extend(n)
        adj.extend(a)
        i += len(n)
    res_adj = [children]
    res_adj.extend(adj)
    return (nodes, res_adj)

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    """
        Wrapper to execute a function stopping it after the given number of seconds.

        Args:
            seconds (float): the maximum number of seconds allowed.

        Raise:
            TimeoutException: if the execution was interrupted
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def utd_to(n1, a1, n2, a2, max_seconds=None, default=-1):
    """
        A wrapper to compute the Unordered Tree edit Distance between two trees (in the node/adjacency form).
        The computation is stopped after a given number of seconds, after which a default value is returned.
        More precisely, 3/4 of the time is devoted to trying to compute `qf.uted.uted.uted_astar`. Then, if
        the computation is not yet over, 1/4 of the time is devoted in trying to compute `qf.uted.uted.uted_constrained`.
        If both fail to succeed in time, the default value is returned

        Args:
            n1 (list): the node part of a node/adjacency representation of a tree.
            a1 (list): the adjacency part of a node/adjacency representation of a tree.
            n2 (list): the node part of a node/adjacency representation of a tree.
            a2 (list): the adjacency part of a node/adjacency representation of a tree.
            max_seconds (float): the maximum number of seconds allowed for the execution, or None if there is no timeout.
            default: the value to be returned on timeout.
        
        Returns:
            the unordered edit distance (as computed by `uted.uted.uted_astar`) between the two trees (n1,a1) and (n2,a2).
    """
    if max_seconds is None:
        return qf.uted.uted.uted_astar(n1, a1, n2, a2)[0]
    try:
        with time_limit(max_seconds * 3 / 4):
            result = qf.uted.uted.uted_astar(n1, a1, n2, a2)[0]
    except TimeoutException:
        logging.info("uted_astar stopped after {} seconds".format(max_seconds * 3 / 4))
        try:
            return default
            with time_limit(max_seconds / 4):
                result = qf.uted.uted.uted_constrained(n1, a1, n2, a2)
        except TimeoutException:
            logging.info("uted_constrained stopped after {} seconds".format(max_seconds / 4))
            result = default
    return result
