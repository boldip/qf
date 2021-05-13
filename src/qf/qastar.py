import networkx as nx
import random
import numpy as np
import sklearn.cluster
import qf.graphs
import uted.uted


def qastarAllPaths(G, target, maxLen, nodeColoring=None):
    """
        Same as allPaths, but it returns a zss.Node instead (the root of the tree). All nodes have the same label, unless
        `nodeColoring` is specified (in which case the value of the map is used).

        Args:
            G: a `networkx.MultiDiGraph`.
            target: a node of G.
            maxLen (int): the maximum length of the paths returned.
            nodeColoring (dict): if not None, it is a dictionary with nodes as keys; the values are used to label the tree nodes.

        Returns:
            a (n,a) dfs-representation.

    """
    n, a = qf.util.dfs_tree(G, target, maxLen)
    if nodeColoring is None:
        return (["x" for x in n], a)
    else:
        return ([nodeColoring[x] for x in n], a)

def qastarTreeDistAlt(G, x, y, maxLen, nodeColoring=None):
    """
        Provides the zss.simple_distance between zssAllPaths(G,x,maxLen,nodeColoring) and zssAllPaths(G,y,maxLen,nodeColoring).
        This function is very inefficient, please use `zssTreeDist` instead.

        Args:
            G: a `networkx.MultiDiGraph`.
            target: a node of G.
            maxLen (int): the maximum length of the paths returned.
            nodeColoring (dict): if not None, it is a dictionary with nodes as keys; the values are used to label the tree nodes.

        Returns: 
            the ZSS (edit) distance between the trees obtained truncating at depth maxLen the universal total graphs of x and y in G.
    """
    nx, ax = qastarAllPaths(G, x, maxLen, nodeColoring)
    ny, ay = qastarAllPaths(G, y, maxLen, nodeColoring)
    return uted.uted.uted_astar(nx, ax, ny, ay)[0]
    
def qastarDistMatrixAlt(G, t, nodeColoring=None):
    """
        Given a graph G and a value t, it computes all the zssAllPaths(G,x,t) trees (for all nodes x of G) and 
        computes all-pairs matrix. The matrix is returned as an np.ndarray, along with the list of nodes (in the order 
        of indices in the matrix) and a map from nodes to indices.
        This function is very inefficient, please use `cachedZssDistMatrix` instead.

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
        d[nodes[i]] = qastarAllPaths(G, nodes[i], t, nodeColoring)
        indices[nodes[i]] = i
    M=np.ndarray((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i,j] = uted.uted.uted_astar(d[nodes[i]][0], d[nodes[i]][1], d[nodes[j]][0], d[nodes[j]][1])
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                 M[i,j] = 0 
            else:
                 M[i,j] = M[j,i]
    return (M, nodes, indices)

def qastarDistMatrix(G, t, nodeColoring=None):
    """
        Given a graph G and a value t, it computes all the zssAllPaths(G,x,t) trees (for all nodes x of G) and 
        computes all-pairs matrix. The matrix is returned as an np.ndarray, along with the list of nodes (in the order 
        of indices in the matrix) and a map from nodes to indices.

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
        d[nodes[i]] = qastarAllPaths(G, nodes[i], t, nodeColoring)
        indices[nodes[i]] = i
    M=np.ndarray((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i,j] = uted.uted.uted_astar(d[nodes[i]][0], d[nodes[i]][1], d[nodes[j]][0], d[nodes[j]][1])[0]
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                 M[i,j] = 0 
            else:
                 M[i,j] = M[j,i]
    return (M, nodes, indices)

def agclust(G, t, num_clusters, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="single"):
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
            M (`numpy.ndarray`): the distance matrix (if None, `cachedZssDistMatrix(G, t, nodeColoring)` is used).
            nodes (list): the list of nodes (used to index M); it must be None exactly when M is None.
            indices (dict): the dictionary from nodes to indices; it must be None exactly when M is None.
            nodeColoring (dict): used to compute the distance matrix (when M is not None).
            linkage_type (str): the linkage type used to compute distances.

        Returns:
            a tuple (clustering, M, nodes, indices):
            - clustering as returned by `sklearn.cluster.AgglomerativeClustering` (labels are stored in the list `clustering.labels_`)
            - M the matrix used for clustering
            - nodes the list of nodes used to index M
            - indices the dictionary from nodes to indices.
   """
    if M is None:
        M, nodes, indices = qastarDistMatrix(G, t, nodeColoring)
    clustering = sklearn.cluster.AgglomerativeClustering(
        affinity="precomputed", linkage=linkage_type, 
        n_clusters=num_clusters, compute_distances=True)
    clustering.fit(M)
    return (clustering, M, nodes, indices)


def agclustVarcl(G, t, minCl, maxCl, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="single"):
    """
        Given a graph G, it computes a clustering (calling `agclust`)
        clustering with a number of clusters varying from minCl (inclusive) to maxCl (exclusive).
        For every clustering the resulting silhouette score is computed (`sklearn.metrics.silhouette_score`).

        Args:
            G: a `networkx.MultiDiGraph`.
            t (int): the truncation depth (used only if M is not None).
            minCl (int): the minimum number of clusters to be produced (inclusive).
            maxCl (int): the maximum number of clusters to be produced (exclusive).
            M (`numpy.ndarray`): the distance matrix (if None, `cachedZssDistMatrix(G, t, nodeColoring)` is used).
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
        M, nodes, indices = qastarDistMatrix(G, t, nodeColoring)
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

def agclustOptcl(G, t, minCl, maxCl, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="average"):
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
            M (`numpy.ndarray`): the distance matrix (if None, `cachedZssDistMatrix(G, t, nodeColoring)` is used).
            nodes (list): the list of nodes (used to index M); it must be None exactly when M is None.
            indices (dict): the dictionary from nodes to indices; it must be None exactly when M is None.
            nodeColoring (dict): used to compute the distance matrix (when M is not None).
            linkage_type (str): the linkage type used to compute distances.

        Returns:
            a tuple (clustering, M, nodes, indices):
            - clustering as returned by `sklearn.cluster.AgglomerativeClustering` (labels are stored in the list `clustering.labels_`)
            - M the matrix used for clustering
            - nodes the list of nodes used to index M
            - indices the dictionary from nodes to indices.
    """
    res, M, nodes, indices = agclustVarcl(G, t, minCl, maxCl, M, nodes, indices, nodeColoring, linkage_type)
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
        
    

