"""
    In this module, the following special representation of trees used in (edist)[https://gitlab.ub.uni-bielefeld.de/bpaassen/python-edit-distances].
    Given a tree of k nodes, visit it in DFS, and assign consecutive number (DFS index) to the nodes you visit. 
    You will have indices 0,1,...,k-1 (where 0 is the root, 1 is its left child, 2 the left child of the latter etc.).
    The tree is represented by a pair (n,a) where:
    
    - n is the *node list*, i.e. the list of its k nodes (in the order of DFS index)
    - a is the *adjacency list*, i.e. a list of k elements, where the i-th element is the list of DFS indices of children of i.

    We call this pair the *node/adjancency representation of the tree*.
    You can do just the same thing for views (of course, you may find the same node many times in the node list).
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

import edist.tree_utils
import networkx as nx
import numpy as np
import sklearn.cluster

import qf.graphs
import qf.uted.uted


def qastar_all_paths(G, target, maxLen, nodeColoring=None):
    """
        Builds the node/adjacency representation (n,a) of the view of `target` in `G` truncated at depth `maxLen`. Before
        returning the pair, the elements of n are mapped as follows: if `nodeColoring` is None, they are all mapped to
        the string "x", otherwise, they are mapped through `nodeColoring`.

        Args:
            G: a `networkx.MultiDiGraph`.
            target: a node of G.
            maxLen (int): the maximum length of the paths returned.
            nodeColoring (dict): if not None, it is a dictionary with nodes as keys; the values are used to label the tree nodes.

        Returns:
            the node/adjacency representation of the view, re-mapped as explained above.

    """
    n, a = qf.util.dfs_tree(G, target, maxLen)
    if nodeColoring is None:
        return (["x" for x in n], a)
    else:
        return ([nodeColoring[x] for x in n], a)


def qastar_dist_matrix(G, t, Msubs=None, nodeColoring=None, max_milliseconds=None, zero=None):
    """
        Given a graph G and a value t, it computes all the `qastar_all_paths(G,x,t)` trees (for all nodes x of G) and 
        computes all-pairs matrix of `uted.uted.uted_astar` distances. The matrix is returned as an np.ndarray, along with the list of nodes (in the order 
        of indices in the matrix) and a map from nodes to indices.

        Args:
            G: a `networkx.MultiDiGraph`.
            t (int): the truncation depth.
            Msubs (`numpy.ndarray`): substitute matrix (see below).
            nodeColoring (dict): if not None, it is a dictionary with nodes as keys; the values are used to label the tree nodes.
            max_milliseconds (int): if not None, it will try to keep the overall computation required within the specified number of milliseconds;
                if the computation of a given entry of the matrix exceeds the time-limit imposed, and `Msubs` is not None, the corresponding entry
                of `Msubs` is used instead.
            zero (dict): if not None, it is a dictionary whose keys are nodes: nodes with the same value are assigned distance zero without any
                further processing.

        Returns:
            a tuple (M, nodes, indices), where
            - M is a `numpy.ndarray` of shape (n,n) (where n is the number of nodes)
            - nodes is a list containing all nodes (exactly once)
            - indices is a dict from nodes to indices.
            The entry M[i,j] is the uted (unordered tree edit) distance between the trucated universal trees at `node[i]` and `node[j]`. 
    """
    nodes = list(G.nodes)
    n = len(nodes)
    d = {}
    indices = {}
    for i in range(n):
        d[nodes[i]] = qastar_all_paths(G, nodes[i], t, nodeColoring)
        indices[nodes[i]] = i
    M=np.ndarray((n, n))
    c=0
    stopped=0
    total_size=n*(n-1)/2 # Number of entries to be computed
    for i in range(n):
        for j in range(i + 1, n):
            if zero is not None and zero[nodes[i]] == zero[nodes[j]]:
                M[i,j] = 0
                continue
            logging.debug("Computing distance from {} [{}] to {} [{}] ({:.3}%)".format(i,nodes[i],j,nodes[j],100*c/total_size))
            logging.debug("Tree 1: {}".format(d[nodes[i]]))
            logging.debug("Tree 2: {}".format(d[nodes[j]]))
            logging.debug("Tree 1 (string): {}".format(edist.tree_utils.tree_to_string(*d[nodes[i]])))
            logging.debug("Tree 2 (string): {}".format(edist.tree_utils.tree_to_string(*d[nodes[j]])))
            if max_milliseconds is None:
                M[i,j] = qf.util.utd_to(d[nodes[i]][0], d[nodes[i]][1], d[nodes[j]][0], d[nodes[j]][1], max_seconds=None)
            else:
                M[i,j] = qf.util.utd_to(d[nodes[i]][0], d[nodes[i]][1], d[nodes[j]][0], d[nodes[j]][1], max_seconds=max_milliseconds / (1000 * total_size))
                if M[i,j] < 0:
                    stopped += 1
                    if Msubs is not None:
                        M[i,j] = Msubs[i,j]
                    logging.info("uted_astar and uted_constrained both stopped, substituted with {}".format(M[i,j]))
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                 M[i,j] = 0 
            else:
                 M[i,j] = M[j,i]
    logging.info("Stopped {:.2f}% of the times".format(100*stopped/total_size))
    return (M, nodes, indices)

def agclust(G, t, num_clusters, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="single", zero=None):
    """
        Given a graph G, it  produces an agglomerative
        clustering with `num_clusters` clusters. The result is returned in the same form as that returned by
        `sklearn.cluster.AgglomerativeClustering`. It also returns the distance matrix used (which is M or
        the one obtained computing all unordered edit distances distances for trees of height t), the 
        list of nodes (in the order of indices in the matrix) and a map from nodes to indices.

        Args:
            G: a `networkx.MultiDiGraph`.
            t (int): the truncation depth (used only if M is not None).
            num_clusters (int): the number of clusters to be produced.
            M (`numpy.ndarray`): the distance matrix (if None, `qastar_dist_matrix(G, t, nodeColoring)` is used).
            nodes (list): the list of nodes (used to index M); it must be None exactly when M is None.
            indices (dict): the dictionary from nodes to indices; it must be None exactly when M is None.
            nodeColoring (dict): used to compute the distance matrix (when M is not None).
            linkage_type (str): the linkage type used to compute distances.
            zero (dict): if not None, it is a dictionary whose keys are nodes: nodes with the same value are assigned distance zero without any
                further processing.

        Returns:
            a tuple (clustering, M, nodes, indices):
            - clustering as returned by `sklearn.cluster.AgglomerativeClustering` (labels are stored in the list `clustering.labels_`)
            - M the matrix used for clustering
            - nodes the list of nodes used to index M
            - indices the dictionary from nodes to indices.
   """
    if M is None:
        M, nodes, indices = qastar_dist_matrix(G, t, nodeColoring, zero=zero)
    clustering = sklearn.cluster.AgglomerativeClustering(
        affinity="precomputed", linkage=linkage_type, 
        n_clusters=num_clusters, compute_distances=True)
    clustering.fit(M)
    return (clustering, M, nodes, indices)


def agclust_varcl(G, t, minCl, maxCl, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="single", zero=None):
    """
        Given a graph G, it computes a clustering (calling `agclust`)
        clustering with a number of clusters varying from minCl (inclusive) to maxCl (exclusive).
        For every clustering the resulting silhouette score is computed (`sklearn.metrics.silhouette_score`).

        Args:
            G: a `networkx.MultiDiGraph`.
            t (int): the truncation depth (used only if M is not None).
            minCl (int): the minimum number of clusters to be produced (inclusive).
            maxCl (int): the maximum number of clusters to be produced (exclusive).
            M (`numpy.ndarray`): the distance matrix (if None, `qastar_dist_matrix(G, t, nodeColoring)` is used).
            nodes (list): the list of nodes (used to index M); it must be None exactly when M is None.
            indices (dict): the dictionary from nodes to indices; it must be None exactly when M is None.
            nodeColoring (dict): used to compute the distance matrix (when M is not None).
            linkage_type (str): the linkage type used to compute distances.
            order_label (str): if not None, every node must have that label as attribute, and the associated values are used
                to sort children in trees.
            zero (dict): if not None, it is a dictionary whose keys are nodes: nodes with the same value are assigned distance zero without any
                further processing.


        Returns:
            The result returned is the same as in `agclust`, but the first component is a dictionary with 
            keys the number of clusters and values are a pair made by the the corresponding clustering and the silhouette score. Note 
            that if for some specific number of clusters the clustering procedure raises an exception, we just avoid
            adding the corresponding result to the dictionary.
    """
    if M is None:
        M, nodes, indices = qastar_dist_matrix(G, t, nodeColoring, zero)
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
            logging.info(type(exc), exc.args, exc)
            pass
    return (res, M, nodes, indices)

def agclust_optcl(G, t, minCl, maxCl, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="single", zero=None):
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
            M (`numpy.ndarray`): the distance matrix (if None, `qastar_dist_matrix(G, t, nodeColoring)` is used).
            nodes (list): the list of nodes (used to index M); it must be None exactly when M is None.
            indices (dict): the dictionary from nodes to indices; it must be None exactly when M is None.
            nodeColoring (dict): used to compute the distance matrix (when M is not None).
            linkage_type (str): the linkage type used to compute distances.
            zero (dict): if not None, it is a dictionary whose keys are nodes: nodes with the same value are assigned distance zero without any
                further processing.

        Returns:
            a tuple (clustering, M, nodes, indices):
            - clustering as returned by `sklearn.cluster.AgglomerativeClustering` (labels are stored in the list `clustering.labels_`)
            - M the matrix used for clustering
            - nodes the list of nodes used to index M
            - indices the dictionary from nodes to indices.
    """
    res, M, nodes, indices = agclust_varcl(G, t, minCl, maxCl, M, nodes, indices, nodeColoring, linkage_type, zero=None)
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
        
    

