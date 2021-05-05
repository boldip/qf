#!/usr/bin/env python
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib as mpl
import random
import collections
from collections import Counter
from sklearn.cluster import DBSCAN
import numpy as np


# Given a graph G, with every arc having a distinct attribute "name", builds and returns the pair (F,B)
# of forward and backward matrices. Columns are indexed by nodes (that are supposed to be named 0,1,...),
# rows are numbered by ordering lexicographically names of arcs.
def incidence(G):
    nodes = list(G.nodes())
    nodes.sort()
    f = np.zeros((G.number_of_edges(), G.number_of_nodes()))
    b = np.zeros((G.number_of_edges(), G.number_of_nodes()))
    m = collections.OrderedDict(sorted({d["label"]: (u,v) for u,v,d in G.edges(data=True)}.items()))
    for i, e in enumerate(m):
        f[i, nodes.index(m[e][0])] = 1
        b[i, nodes.index(m[e][1])] = 1
    return f, b

# Given a graph G, with every arc having a distinct attribute "name", builds the adjacency matrix of G. 
def adjacence(G):
    M = np.zeros((G.number_of_edges(), G.number_of_edges()))
    l = list(G.edges())
    m = collections.OrderedDict(sorted({d["label"]: (u,v) for u,v,d in G.edges(data=True)}.items()))
    for i, e in enumerate(m):
        M[m[e][0], m[e][1]] += 1
    return M

# The map must be a pair (map for nodes, map for arcs). It checks that the map is compatible.
# It returns the pair 
def matrix_of(G, B, map):
    node_map = map[0]
    edge_map = map[1]
    gnodes = list(G.nodes())
    gnodes.sort()
    bnodes = list(B.nodes())
    bnodes.sort()
    node_matrix = np.zeros((G.number_of_nodes(), B.number_of_nodes()))
    edge_matrix = np.zeros((G.number_of_edges(), B.number_of_edges()))
    for k,v in node_map.items():
        node_matrix[gnodes.index(k),bnodes.index(v)] = 1
    mG = collections.OrderedDict(sorted({d["label"]: (u,v) for u,v,d in G.edges(data=True)}.items()))    
    mB = collections.OrderedDict(sorted({d["label"]: (u,v) for u,v,d in B.edges(data=True)}.items()))    
    for gArcName,(u,v) in mG.items():
        i = list(mG.keys()).index(gArcName)
        bArcName = edge_map[gArcName]  
        j = list(mB.keys()).index(bArcName)
        edge_matrix[i,j] = 1
        # check correctness
        if node_map[u] != mB[bArcName][0] or node_map[v] != mB[bArcName][1]:
            print("Wrong map: sends ", gArcName, "->", bArcName, ", but sources: ", node_map[u], "->", mB[bArcName][0], ", targets: ",  node_map[v], "->", mB[bArcName][1])
    return (node_matrix, edge_matrix)


######
# Checking functions 


def delta(n):
    M = np.zeros((n, n*n))
    for i in range(n):
        M[i][i * (n + 1)] = 1
    return M


def check_map(G, B, map):
    (f_B, b_B) = incidence(B)
    (f_G, b_G) = incidence(G)
    m_n, m_e = matrix_of(G, B, map)
    print(np.array_equal(np.matmul(f_G, m_n), np.matmul(m_e, f_B)))
    return np.array_equal(np.matmul(f_G, m_n), np.matmul(m_e, f_B)) and np.array_equal(np.matmul(b_G, m_n), np.matmul(m_e, b_B))
    

def check_fib(G, B, map):
    (f_B, b_B) = incidence(B)
    (f_G, b_G) = incidence(G)
    m_n, m_e = matrix_of(G, B, map)
    M = np.matmul(delta(G.number_of_edges()), np.kron(m_e, b_G))
    N = np.matmul(np.matmul(np.matmul(delta(B.number_of_nodes()), np.kron(np.transpose(b_B), np.transpose(m_n))), np.kron(np.transpose(m_e),np.transpose(b_G))), np.transpose(delta(G.number_of_edges())))
    print(np.matmul(M, np.transpose(M)))
    print(N)
    print(np.matmul(N, np.transpose(N)))
    return (np.array_equal(np.matmul(M, np.transpose(M)), np.identity(G.size())),
            np.array_equal(np.matmul(N, np.transpose(N)), np.identity(B.size())))


