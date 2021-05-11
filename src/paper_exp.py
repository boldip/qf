#!/usr/bin/env python

import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
import qf.cc
import qf.graphs
import qf.util
import qf.qzss
import qf.zssexp
import qf.morph
import sys
import argparse

if __name__ != "__main__":
	exit()

## Argument parsing
argparser = argparse.ArgumentParser(description="Perform experiments on a single dataset")

argparser.add_argument("graph_filename", type=str,
                       help="Graph to be processed (for synthetic: use :n:noise or :n:add:del)")
argparser.add_argument("--ground", type=str,
                       help="File with ground truth (otherwise, the coarsest equitable partition will be used)")
argparser.add_argument("--depth", type=int, default=2,
                       help="Tree cut depth")
argparser.add_argument("--prec", type=int, default=10,
                       help="Precision in parameter grids")
argparser.add_argument("--graph_skip_header", type=bool, default=True,
                       help="Skip headers while reading graph file")
argparser.add_argument("--graph_separator", type=str, default="\t",
                       help="Separator used in graph file")
argparser.add_argument("--graph_is_dense", type=bool, default=False,
                       help="Whether the graph is specified in the dense format")
argparser.add_argument("--ground_skip_header", type=bool, default=True,
                       help="Skip headers while reading ground truth file")
argparser.add_argument("--ground_separator", type=str, default="\t",
                       help="Separator used in ground truth file")
args = argparser.parse_args() 
    
## Read file

if args.graph_filename.startswith(":"):
    if args.ground is not None:
        parser.error("Cannot specify ground truth for synthetic datasets")
    spl = args.graph_filename.split(":")
    if len(spl) < 3 or len(spl) > 4:
        parser.error("Wrong filename specification {}".format(args.graph_filename))
    try:
        n = int(spl[1])
        if len(spl) == 3:
            noise = int(spl[2])
        else:
            noise = -1
            add_noise = int(spl[2])
            del_noise = int(spl[3])
    except ValueError:
        parser.error("Wrong synthetic dataset specification {}".format(args.graph_filename))

    dataset_name = "Synthetic"
    Gideal = qf.zssexp.getFibrationRichGraph(n)
    gt = qf.cc.cardon_crochemore(Gideal)
    gtn = len(set(gt.values()))
    if noise >= 0:
        G = qf.graphs.scramble(Gideal, nAdd=0, nDel=0, nScr=noise)
    else:
        G = qf.graphs.scramble(Gideal, nAdd=add_noise, nDel=del_noise)
else:
    dataset_name = "{} [{}]".format(args.graph_filename, args.ground)
    G = qf.util.readGraph(args.graph_filename, skipHeader=args.graph_skip_header, separator=args.graph_separator, dense=args.graph_is_dense)
    if args.ground is None:
        gt = qf.cc.cardon_crochemore(G)
    else:
        gt = qf.util.readLabel(args.ground, skipHeader=args.ground_skip_header, separator=args.ground_separator)
    gtn = len(set(gt.values()))

depth = args.depth
prec = args.prec
results = {}


# Build nodes and indices
nodes=list(G.nodes)
indices = {}
for i in range(len(nodes)):
    indices[nodes[i]] = i


#Compute Cardon-Crochemore
cc=qf.cc.cardon_crochemore(G)
ccn =  len(set(cc.values()))
ccnmi = qf.util.nmi(gt, cc)
results["Cardon-Crochemore"]=(ccn,ccnmi)


#Compute agglomerative clustering on the pure ZSS matrix
# ZSS matrix
for linkage_type in ["single", "average", "complete"]:
    M, nodes, indices = qf.qzss.cachedZssDistMatrix(G, depth)        
    nM = M/sum(sum(M))

    # Agglomerative clustering
    c, _M, nodes, indices = qf.qzss.agclustOptcl(G, depth, nM, nodes, indices, linkage_type=linkage_type)
    bestc = qf.qzss.agclust2dict(c, _M, nodes, indices)
    bestcn = len(set(bestc.values()))
    bestcnmi = qf.util.nmi(gt, bestc)
    description="Agglomerative (linkage={})".format(linkage_type)
    results[description]=(bestcn,bestcnmi)

    # Doing further steps
    B, xi = qf.morph.qf_build(G, bestc, verbose=False)
    Gp, xip = qf.morph.repair(xi, G, B, verbose=False)
    ccp = qf.cc.cardon_crochemore(Gp)
    ccpn =  len(set(ccp.values()))
    ccpnmi = qf.util.nmi(gt, ccp)
    results["Final (aggl., linkage={})"]=(ccpn,ccpnmi)

    #Compute agglomerative clustering on the pure ZSS matrix with exact number of clusters

    c, _M, nodes, indices = qf.qzss.agclust(G, depth, gtn, nM, nodes, indices, linkage_type=linkage_type)
    bestcex = qf.qzss.agclust2dict(c, _M, nodes, indices)
    bestcexn = len(set(bestcex.values()))
    bestcexnmi = qf.util.nmi(gt, bestcex)
    description="Agglomerative exact (linkage={}) [*]".format(linkage_type)
    results[description]=(bestcexn,bestcexnmi)

# Simple graph
H = qf.graphs.to_simple(G)
n = H.number_of_nodes()
node_list = [x for x in H.nodes]

# ## KATZ Centrality
steps=prec
min_range=0.01
max_range=0.99
km = KMeans(gtn)
xs = []
ys = []
for a in range(steps):
    alpha = min_range + a * (max_range - min_range) / steps
    xs.append(alpha)
    try:
        katz = nx.katz_centrality(H, alpha)
    except:
        break
    km.fit(np.reshape([katz[x] for x in H.nodes],(-1,1)))
    km_col = {node_list[i]: km.labels_[i] for i in range(n)}
    ys.append(qf.util.nmi(km_col,gt))
results["Katz (best) [*]"]=(gtn,max(ys))

# ## PageRank Centrality
steps = prec
min_range=0.01
max_range=0.99
km = KMeans(gtn)
xs = []
ys = []
for a in range(steps):
    alpha = min_range + a * (max_range - min_range) / steps
    xs.append(alpha)
    pr = nx.pagerank(H, alpha)
    km.fit(np.reshape([pr[x] for x in H.nodes],(-1,1)))
    km_col = {node_list[i]: km.labels_[i] for i in range(n)}
    ys.append(qf.util.nmi(km_col,gt))
results["PageRank (best) [*]"]=(gtn,max(ys))

# ## Closeness Centrality
closeness = nx.closeness_centrality(H)
km.fit(np.reshape([closeness[x] for x in H.nodes],(-1,1)))
km_col = {node_list[i]: km.labels_[i] for i in range(n)}
nmi = qf.util.nmi(km_col,gt)
results["Closeness [*]"]=(gtn,nmi)


# ## Eigenvector Centrality
eigen = nx.eigenvector_centrality(H)
km.fit(np.reshape([eigen[x] for x in H.nodes],(-1,1)))
km_col = {node_list[i]: km.labels_[i] for i in range(n)}
nmi = qf.util.nmi(km_col,gt)
results["Eigenvector [*]"]=(gtn,nmi)

# ## Betweenness centrality
betweenness = nx.betweenness_centrality(H)
km.fit(np.reshape([betweenness[x] for x in H.nodes],(-1,1)))
km_col = {node_list[i]: km.labels_[i] for i in range(n)}
nmi = qf.util.nmi(km_col,gt)
results["Betweenness [*]"]=(gtn,nmi)

# ## Harmonic Centrality
harmonic = nx.harmonic_centrality(H)
km.fit(np.reshape([harmonic[x] for x in H.nodes],(-1,1)))
km_col = {node_list[i]: km.labels_[i] for i in range(n)}
nmi = qf.util.nmi(km_col,gt)
results["Harmonic [*]"]=(gtn,nmi)

# ## Trophic level Centrality
try:
    trophic = nx.trophic_levels(H)
    km.fit(np.reshape([trophic[x] for x in H.nodes],(-1,1)))
    km_col = {node_list[i]: km.labels_[i] for i in range(n)}
    nmi = qf.util.nmi(km_col,gt)
    results["Trophic levels [*]"]=(gtn,nmi)
except:
    pass

# ### All results

print("*** Dataset: {}".format(dataset_name))
print("Nodes: {}\nArcs: {}".format(G.number_of_nodes(), G.number_of_edges()))
print("Ground truth: #classes={}\n".format(gtn))
sorted_keys = sorted(results.keys(), key=lambda x: results[x][1], reverse=True)  
for k in sorted_keys:
    print("{:40}\t{:.5}\t\t{:2}".format(k, results[k][1], results[k][0]))
print("\n[*] means that the ground truth size is used")

