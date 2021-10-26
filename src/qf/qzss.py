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

import networkx as nx
import numpy as np
import sklearn.cluster
import zss

import qf.graphs


def all_paths(G, target, maxLen):
    """
        Returns an iterator for all paths in G of length <=maxLen with given target. Any such path will be returned
        as a list [x_1,k_1,...,x_n,k_n,x_{n+1}] where x_{n+1}=target, 0<=n<=maxLen, (x_i,x_{i+1}) is
        an arc with key k_i (for all i=1,...,n).

        Args:
            G: a `networkx.MultiDiGraph`.
            target: a node of G.
            maxLen (int): the maximum length of the paths returned.

        Returns:
            an iterator that returns lists; each returned list will be a non-empty sequence alteranting nodes and arcs of G
            (starting and ending with a node, the last node will always be target) and containing no more than maxLen arcs.
            Every arc will have source equal to the node immediately preceding it, and target equal to the node immediately
            following it. All such lists will be returned, and none of them will be returned more than once. 

    """
    if maxLen > 0:
        for s,t,d in G.in_edges(target, data=True):
            for p in all_paths(G, s, maxLen - 1):
                yield p + [d["label"], target]
    yield [target]

def in_tree(G, target, maxLen):
    """
        Returns a (simple) multidigraph whose nodes are the paths as returned by all_paths(G, target, maxLen), 
        and with an arc from [x_1,k_1,...,x_n,k_n,x_{n+1}] to [x_2,k_2,...,x_n,k_n,x_{n+1}] for all n>0.

        Args:
            G: a `networkx.MultiDiGraph`.
            target: a node of G.
            maxLen (int): the maximum length of the paths returned.

        Returns:
            see above. The graph is the universal total graph of target truncated at depth
    """
    Gres = nx.MultiDiGraph()
    if maxLen == 0:
        Gres.add_node(str([target]))
        return Gres
    for p in all_paths(G, target, maxLen):
        if len(p)>1 and not Gres.has_edge(str(p), str(p[2:])):
            Gres.add_edge(str(p), str(p[2:]))
    return Gres


def zss_all_paths(G, target, maxLen, nodeColoring=None):
    """
        Same as all_paths, but it returns a zss.Node instead (the root of the tree). All nodes have the same label, unless
        `nodeColoring` is specified (in which case the value of the map is used).

        Args:
            G: a `networkx.MultiDiGraph`.
            target: a node of G.
            maxLen (int): the maximum length of the paths returned.
            nodeColoring (dict): if not None, it is a dictionary with nodes as keys; the values are used to label the tree nodes.

        Returns:
            a zss.Node that is the root of a tree isomorphic to the universal total graph of G at target, truncated at
            maxLen.

    """
    if nodeColoring is None:
        node = zss.Node("x")
    else:
        node = zss.Node(nodeColoring[target])
    if maxLen == 0:
        return node  
    res = node
    for s,t,k in G.in_edges(target, keys=True):
        res = res.addkid(zss_all_paths(G, s, maxLen - 1, nodeColoring))
    return res

def zss_tree_dist_alt(G, x, y, maxLen, nodeColoring=None):
    """
        Provides the zss.simple_distance between zss_all_paths(G,x,maxLen,nodeColoring) and zss_all_paths(G,y,maxLen,nodeColoring).
        This function is very inefficient, please use `zss_tree_dist` instead.

        Args:
            G: a `networkx.MultiDiGraph`.
            target: a node of G.
            maxLen (int): the maximum length of the paths returned.
            nodeColoring (dict): if not None, it is a dictionary with nodes as keys; the values are used to label the tree nodes.

        Returns: 
            the ZSS (edit) distance between the trees obtained truncating at depth maxLen the universal total graphs of x and y in G.
    """
    return zss.simple_distance(zss_all_paths(G, x, maxLen, nodeColoring), zss_all_paths(G, y, maxLen, nodeColoring))
    
def cached_zss_dist_matrix_alt(G, t, nodeColoring=None):
    """
        Given a graph G and a value t, it computes all the zss_all_paths(G,x,t) trees (for all nodes x of G) and 
        computes all-pairs matrix. The matrix is returned as an np.ndarray, along with the list of nodes (in the order 
        of indices in the matrix) and a map from nodes to indices.
        This function is very inefficient, please use `cached_zss_dist_matrix` instead.

        Args:
            G: a `networkx.MultiDiGraph`.
            t (int): the truncation depth.
            nodeColoring (dict): if not None, it is a dictionary with nodes as keys; the values are used to label the tree nodes.

        Returns:
            a tuple (M, nodes, indices), where
            - M is a `numpy.ndarray` of shape (n,n) (where n is the number of nodes)
            - nodes is a list containing all nodes (exactly once)
            - indices is a dict from nodes to indices.
            The entry M[i,j] is the ZSS (tree edit) distance
            between the trucated universal trees at `node[i]` and `node[j]`. 
    """
    nodes = list(G.nodes)
    n = len(nodes)
    d = {}
    indices = {}
    for i in range(n):
        d[nodes[i]] = zss_all_paths(G, nodes[i], t, nodeColoring)
        indices[nodes[i]] = i
    M=np.ndarray((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i,j] = zss.simple_distance(d[nodes[i]], d[nodes[j]])
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                 M[i,j] = 0 
            else:
                 M[i,j] = M[j,i]
    return (M, nodes, indices)

def cached_zss_dist_matrix(G, t, nodeColoring=None, order_label=None):
    """
        Given a graph G and a value t, it computes all the zss_all_paths(G,x,t) trees (for all nodes x of G) and 
        computes all-pairs matrix. The matrix is returned as an np.ndarray, along with the list of nodes (in the order 
        of indices in the matrix) and a map from nodes to indices.

        Args:
            G: a `networkx.MultiDiGraph`.
            t (int): the truncation depth.
            nodeColoring (dict): if not None, it is a dictionary with nodes as keys; the values are used to label the tree nodes.
            order_label (str): if not None, every node must have that label as attribute, and the associated values are used
                to sort children in trees.

        Returns:
            a tuple (M, nodes, indices), where
            - M is a `numpy.ndarray` of shape (n,n) (where n is the number of nodes)
            - nodes is a list containing all nodes (exactly once)
            - indices is a dict from nodes to indices.
            The entry M[i,j] is the ZSS (tree edit) distance
            between the trucated universal trees at `node[i]` and `node[j]`. 
    """
    nodes = list(G.nodes)
    n = len(nodes)
    d = {}
    indices = {}
    for i in range(n):
        d[nodes[i]] = SpecialNode(G, nodes[i], t, nodeColoring, order_label)
        indices[nodes[i]] = i
    M=np.ndarray((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i,j] = zss.simple_distance(d[nodes[i]], d[nodes[j]])
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                 M[i,j] = 0 
            else:
                 M[i,j] = M[j,i]
    return (M, nodes, indices)

def agclust(G, t, num_clusters, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="single", order_label=None):
    """
        Given a graph G, it  produces an agglomerative
        clustering with `num_clusters` clusters. The result is returned in the same form as that returned by
        `sklearn.cluster.AgglomerativeClustering`. It also returns the distance matrix used (which is M or
        the one obtained computing all zss distances for trees of height t), the 
        list of nodes (in the order of indices in the matrix) and a map from nodes to indices.

        Args:
            G: a `networkx.MultiDiGraph`.
            t (int): the truncation depth (used only if M is not None).
            num_clusters (int): the number of clusters to be produced.
            M (`numpy.ndarray`): the distance matrix (if None, `cached_zss_dist_matrix(G, t, nodeColoring)` is used).
            nodes (list): the list of nodes (used to index M); it must be None exactly when M is None.
            indices (dict): the dictionary from nodes to indices; it must be None exactly when M is None.
            nodeColoring (dict): used to compute the distance matrix (when M is not None).
            linkage_type (str): the linkage type used to compute distances.
            order_label (str): if not None, every node must have that label as attribute, and the associated values are used
                to sort children in trees.

        Returns:
            a tuple (clustering, M, nodes, indices):
            - clustering as returned by `sklearn.cluster.AgglomerativeClustering` (labels are stored in the list `clustering.labels_`)
            - M the matrix used for clustering
            - nodes the list of nodes used to index M
            - indices the dictionary from nodes to indices.
   """
    if M is None:
        M, nodes, indices = cached_zss_dist_matrix(G, t, nodeColoring, order_label)
    clustering = sklearn.cluster.AgglomerativeClustering(
        affinity="precomputed", linkage=linkage_type, 
        n_clusters=num_clusters, compute_distances=True)
    clustering.fit(M)
    return (clustering, M, nodes, indices)


def agclust_varcl(G, t, minCl, maxCl, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="single", order_label=None):
    """
        Given a graph G, it computes a clustering (calling `agclust`)
        clustering with a number of clusters varying from minCl (inclusive) to maxCl (exclusive).
        For every clustering the resulting silhouette score is computed (`sklearn.metrics.silhouette_score`).

        Args:
            G: a `networkx.MultiDiGraph`.
            t (int): the truncation depth (used only if M is not None).
            minCl (int): the minimum number of clusters to be produced (inclusive).
            maxCl (int): the maximum number of clusters to be produced (exclusive).
            M (`numpy.ndarray`): the distance matrix (if None, `cached_zss_dist_matrix(G, t, nodeColoring)` is used).
            nodes (list): the list of nodes (used to index M); it must be None exactly when M is None.
            indices (dict): the dictionary from nodes to indices; it must be None exactly when M is None.
            nodeColoring (dict): used to compute the distance matrix (when M is not None).
            linkage_type (str): the linkage type used to compute distances.
            order_label (str): if not None, every node must have that label as attribute, and the associated values are used
                to sort children in trees.

        Returns:
            The result returned is the same as in `agclust`, but the first component is a dictionary with 
            keys the number of clusters and values are a pair made by the the corresponding clustering and the silhouette score. Note 
            that if for some specific number of clusters the clustering procedure raises an exception, we just avoid
            adding the corresponding result to the dictionary.
    """
    if M is None:
        M, nodes, indices = cached_zss_dist_matrix(G, t, nodeColoring, order_label)
    res = {}
    for cl in range(minCl, maxCl):
        try:
            clustering = sklearn.cluster.AgglomerativeClustering(
                affinity="precomputed", linkage=linkage_type, 
                n_clusters=cl, compute_distances=True)
            clustering.fit(M)
            silhouette = sklearn.metrics.silhouette_score(M, clustering.labels_, metric="precomputed")
            res[cl]=(clustering, silhouette)
        except Exception as exc:
            print(type(exc))
            print(exc.args)
            print(exc)
            pass
    return (res, M, nodes, indices)

def agclust_optcl(G, t, minCl, maxCl, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="average", order_label=None):
    """
        Given a graph G, it computes a clustering (calling `agclust`)
        clustering with a number of clusters varying from minCl (inclusive) to maxCl (exclusive).
        For every clustering the resulting silhouette score is computed (`sklearn.metrics.silhouette_score`),
        and the first clustering producing the maximal silhouette is returned (in the same form as in
        `agclust`.

        Args:
            G: a `networkx.MultiDiGraph`.
            t (int): the truncation depth (used only if M is not None).
            minCl (int): the minimum number of clusters to be produced (inclusive).
            maxCl (int): the maximum number of clusters to be produced (exclusive).
            M (`numpy.ndarray`): the distance matrix (if None, `cached_zss_dist_matrix(G, t, nodeColoring)` is used).
            nodes (list): the list of nodes (used to index M); it must be None exactly when M is None.
            indices (dict): the dictionary from nodes to indices; it must be None exactly when M is None.
            nodeColoring (dict): used to compute the distance matrix (when M is not None).
            linkage_type (str): the linkage type used to compute distances.
            order_label (str): if not None, every node must have that label as attribute, and the associated values are used
                to sort children in trees.

        Returns:
            a tuple (clustering, M, nodes, indices):
            - clustering as returned by `sklearn.cluster.AgglomerativeClustering` (labels are stored in the list `clustering.labels_`)
            - M the matrix used for clustering
            - nodes the list of nodes used to index M
            - indices the dictionary from nodes to indices.
    """
    res, M, nodes, indices = agclust_varcl(G, t, minCl, maxCl, M, nodes, indices, nodeColoring, linkage_type, order_label)
    maxsilhouette=max([v[1] for v in res.values()])
    for optCl in res.keys():
        if res[optCl][1]==maxsilhouette:
            break
    return (res[optCl][0], M, nodes, indices)

def agclust2dict(clustering, M, nodes, indices):
    """
        Given the results of agclust, produces a labelling for the nodes of G (a map from nodes to clusters).

        Args:
            clustering: a clustering as returned by `sklearn.cluster.AgglomerativeClustering` (labels are stored in the list `clustering.labels_`)
            M (`numpy.ndarray`): the distance matrix used to compute the clustering (ignored by this function).
            nodes (list): the list of nodes (used to index M); it must be None exactly when M is None.
            indices (dict): the dictionary from nodes to indices; it must be None exactly when M is None.

        Returns:
            a dictionary whose keys are nodes, and where two keys are associated the same value iff they belong to the same cluster.
    """
    return {x: clustering.labels_[indices[x]] for x in nodes}
        
    
class SpecialNode(object):
    """
        An alternative implementation of a zss.Node that does not need to actually unfold the paths.
    """

    def __init__(self, G, x, depth, nodeColoring, order_label):
        """
            Creates a node of a truncated universal total graph.

            Args:
                G: the graph (a `networkx.MultiDiGraph`)
                x: the node (the root of the universal total graph)
                depth (int): the depth of the view.
                nodeColoring (dict): if present, it is a map from nodes whose values are used to color the tree.
                order_label (str): if not None, every node must have that label as attribute, and the associated values are used
                    to sort children in trees.
        """
        self.G = G
        self.x = x
        self.depth = depth
        if nodeColoring is None:
            self.label = "x"
        else:
            self.label = nodeColoring[x]
        self.children = []
        if depth > 0:
            chl = []
            for s,t,d in G.in_edges(x, data=True):
                chl.append(s)
            if order_label is not None:
                d = nx.get_node_attributes(G, order_label)
                chl = sorted(chl, key=lambda node: d[node])
            for child in chl:
                self.children.append(SpecialNode(G, child, depth - 1, nodeColoring, order_label))

    @staticmethod
    def get_children(node):
        return node.children
            
    @staticmethod
    def get_label(node):
        return node.label

    def recprint(self, level):
        for i in range(level):
            print("\t", end="")
        print(self.x)
        for c in self.children:
            c.recprint(level + 1)

def zss_tree_dist(G, x, y, maxLen, nodeColoring=None):
    """
        Provides the zss.simple_distance between zss_all_paths(G,x,maxLen,nodeColoring) and zss_all_paths(G,y,maxLen,nodeColoring).

        Args:
            G: a `networkx.MultiDiGraph`.
            target: a node of G.
            maxLen (int): the maximum length of the paths returned.
            nodeColoring (dict): if not None, it is a dictionary with nodes as keys; the values are used to label the tree nodes.

        Returns: 
            the ZSS (edit) distance between the trees obtained truncating at depth maxLen the universal total graphs of x and y in G.
    """
    return zss.simple_distance(SpecialNode(G, x, maxLen, nodeColoring, None), SpecialNode(G, y, maxLen, nodeColoring, None))

def katz_preorder(G, order_label):
    """
        Compute Katz centrality on (the simple version of) G, and add its value as a node attribute with the given label.

        Args:
            G: the graph involved.
            order_label: the label to be used for the new node attribute.
    """
    nx.set_node_attributes(G, nx.katz_centrality(qf.graphs.to_simple(G)), order_label)

     
    