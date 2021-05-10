import networkx as nx
import random
import numpy as np
import sklearn.cluster
import zss

def allPaths(G, target, maxLen):
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
            for p in allPaths(G, s, maxLen - 1):
                yield p + [d["label"], target]
    yield [target]

def inTree(G, target, maxLen):
    """
        Returns a (simple) multidigraph whose nodes are the paths as returned by allPaths(G, target, maxLen), 
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
    for p in allPaths(G, target, maxLen):
        if len(p)>1 and not Gres.has_edge(str(p), str(p[2:])):
            Gres.add_edge(str(p), str(p[2:]))
    return Gres


def zssAllPaths(G, target, maxLen, nodeColoring=None):
    """
        Same as allPaths, but it returns a zss.Node instead (the root of the tree). All nodes have the same label ("x").

        Args:
            G: a `networkx.MultiDiGraph`.
            target: a node of G.
            maxLen (int): the maximum length of the paths returned.

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
        res = res.addkid(zssAllPaths(G, s, maxLen - 1, nodeColoring))
    return res

# Provides the zss.simple_distance between zssAllPaths(G,x,maxLen) and zssAllPaths(G,y,maxLen)
def zssTreeDist(G, x, y, maxLen, nodeColoring=None):
    return zss.simple_distance(zssAllPaths(G, x, maxLen, nodeColoring), zssAllPaths(G, y, maxLen, nodeColoring))
    

# Given a graph G and a value t, it computes all the zssAllPaths(G,x,t) trees (for all nodes x of G) and 
# computes all-pairs matrix. The matrix is returned as an np.ndarray, along with the list of nodes (in the order 
# of indices in the matrix) and a map from nodes to indices.
def cachedZssDistMatrix(G, t, nodeColoring=None):
    nodes = list(G.nodes)
    n = len(nodes)
    d = {}
    indices = {}
    for i in range(n):
        d[nodes[i]] = zssAllPaths(G, nodes[i], t, nodeColoring)
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


# Given a graph G, it computes all zss distances for trees of height t, and produces an agglomerative
# clustering with num_clusters clusters. The result is returned in the same form as that returned by
# sklearn.cluster.AgglomerativeClustering. It also returns the distance matrix used and the 
# list of nodes (in the order of indices in the matrix) and a map from nodes to indices.
def agclust(G, t, num_clusters, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="average"):
    if M is None:
        M, nodes, indices = cachedZssDistMatrix(G, t, nodeColoring)
    clustering = sklearn.cluster.AgglomerativeClustering(
        affinity="precomputed", linkage=linkage_type, 
        n_clusters=num_clusters, compute_distances=True)
    clustering.fit(M)
    return (clustering, M, nodes, indices)


# Given a graph G, it computes all zss distances for trees of height t, and produces an agglomerative
# clustering with a number of clusters varying from minCl (inclusive) to maxCl (exclusive).
# For every clustering the resulting silhouette score is computed.
# If M is provided, it must be provided along with nodes and indices and that matrix is used instead.
# The result returned is the same as in agclust, but the first component is a dictionary with 
# keys the number of clusters and values is a pair made by the the corresponding clustering and the silhouette.
def agclustVarcl(G, t, minCl, maxCl, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="average"):
    if M is None:
        M, nodes, indices = cachedZssDistMatrix(G, t, nodeColoring)
    res = {}
    for cl in range(minCl, maxCl):
        clustering = sklearn.cluster.AgglomerativeClustering(
            affinity="precomputed", linkage=linkage_type, 
            n_clusters=cl, compute_distances=True)
        clustering.fit(M)
        silhouette = sklearn.metrics.silhouette_score(M, clustering.labels_, metric="precomputed")
        res[cl]=(clustering, silhouette)
    return (res, M, nodes, indices)

# Given a graph G, it computes all zss distances for trees of height t, and produces an agglomerative
# clustering with a number of clusters computed as follows: all numbers between 2 and the number of nodes of
# G is tried, every time the silhouette is computed, and at the end the 
# cluster with maximum silhouette is used.
# If M is provided, it must be provided along with nodes and indices and that matrix is used instead.
# The result returned is the same as in agclust.
def agclustOptcl(G, t, M=None, nodes=None, indices=None, nodeColoring=None, linkage_type="average"):
    minCl = min(4, G.number_of_nodes())
    res, M, nodes, indices = agclustVarcl(G, t, minCl, G.number_of_nodes(), M, nodes, indices, nodeColoring, linkage_type)
    maxsilhouette=max([v[1] for v in res.values()])
    for optCl in res.keys():
        if res[optCl][1]==maxsilhouette:
            break
    return (res[optCl][0], M, nodes, indices)

# Given the results of agclust, produces a labelling for the nodes of G (a map from nodes to clusters).
def agclust2dict(clustering, M, nodes, indices):
    return {x: clustering.labels_[indices[x]] for x in nodes}
        
    
    
    
    