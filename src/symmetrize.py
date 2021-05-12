#!/usr/bin/env python

import argparse
import logging
import random
import sys
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython import get_ipython
from networkx.drawing.nx_agraph import write_dot
from sklearn.cluster import DBSCAN, KMeans

import qf.cc
import qf.graphs
import qf.morph
import qf.qzss
import qf.util
import qf.zssexp

if __name__ != "__main__":
	exit()

## Argument parsing
argparser = argparse.ArgumentParser(description="Perform experiments on a single dataset")

argparser.add_argument("graph_filename", type=str,
                       help="Graph to be processed (for synthetic: use :n:noise or :n:add:del)")
argparser.add_argument("output_basename", type=str,
                       help="Output basename")
argparser.add_argument("--ground", type=str,
                       help="File with ground truth (otherwise, the coarsest equitable partition will be used)")
argparser.add_argument("--coord", type=str,
                       help="File with the coordinates to be used for the display")
argparser.add_argument("--depth", type=int, default=2,
                       help="Tree cut depth")
#argparser.add_argument("--prec", type=int, default=10,
#                       help="Precision in parameter grids")
argparser.add_argument("--graph_skip_header", action="store_false",
                       help="Skip headers while reading graph file")
argparser.add_argument("--graph_separator", type=str, default="\t",
                       help="Separator used in graph file")
argparser.add_argument("--graph_is_dense", action="store_true",
                       help="Whether the graph is specified in the dense format")
argparser.add_argument("--coord_skip_header", action="store_false",
                       help="Skip headers while reading coord file")
argparser.add_argument("--coord_separator", type=str, default=" ",
                       help="Separator used in coord file")
argparser.add_argument("--ground_skip_header", action="store_false",
                       help="Skip headers while reading ground truth file")
argparser.add_argument("--ground_separator", type=str, default="\t",
                       help="Separator used in ground truth file")
argparser.add_argument("--katz", action="store_true",
                       help="Order children in trees using Katz centrality")
args = argparser.parse_args() 
    
## Read file

if args.graph_filename.startswith(":"):
    logging.info("Creating synthetic graph")
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
    logging.info("Reading graph file")
    dataset_name = "{} [{}]".format(args.graph_filename, args.ground)
    coords = None
    if args.coord is not None:
        coords = qf.util.readCoordinates(args.coord, skipHeader=args.coord_skip_header, separator=args.coord_separator)    
    G = qf.util.readGraph(args.graph_filename, skipHeader=args.graph_skip_header, separator=args.graph_separator, dense=args.graph_is_dense, coordinates=coords)
    if args.ground is None:
        raise Exception("No ground truth available for real graph")
    else:
        gt = qf.util.readLabel(args.ground, skipHeader=args.ground_skip_header, separator=args.ground_separator)
    gtn = len(set(gt.values()))

depth = args.depth
results = {}

# Compute Katz if needed
order_label = None
if args.katz:
    qf.qzss.katz_preorder(G, "katz")
    order_label = "katz"

# Build nodes and indices
nodes=list(G.nodes)
indices = {}
for i in range(len(nodes)):
    indices[nodes[i]] = i


#Compute Cardon-Crochemore
logging.info("Running Cardon-Crochemore")
cc=qf.cc.cardon_crochemore(G)
ccn =  len(set(cc.values()))
ccnmi = qf.util.nmi(gt, cc)
results["Cardon-Crochemore"]=(ccn,ccnmi)


#Compute agglomerative clustering on the pure ZSS matrix
# ZSS matrix
logging.info("Running Agglomerative Clustering")
for linkage_type in ["single"]:
    M, nodes, indices = qf.qzss.cachedZssDistMatrix(G, depth, order_label=order_label)        
    nM = M/sum(sum(M))

    # Agglomerative clustering
    c, _M, nodes, indices = qf.qzss.agclustOptcl(G, depth, 2, len(nodes), nM, nodes, indices, linkage_type=linkage_type, order_label=order_label)
    bestc = qf.qzss.agclust2dict(c, _M, nodes, indices)
    bestcn = len(set(bestc.values()))
    bestcnmi = qf.util.nmi(gt, bestc)
    description="Agglomerative (linkage={})".format(linkage_type)
    results[description]=(bestcn,bestcnmi)


# Save graph files
logging.info("Writing dot/png files")
qf.graphs.save(G, args.output_basename + "-orig-cc.dot", args.output_basename + "-orig-cc.png", colors=cc)
qf.graphs.save(G, args.output_basename + "-orig-aggl.dot", args.output_basename + "-orig-aggl.png", colors=bestc)
qf.graphs.save(G, args.output_basename + "-orig-gt.dot", args.output_basename + "-orig-gt.png", colors=gt)


# Completion
logging.info("Building quasi-fibration and repairing")
B, xi = qf.morph.qf_build(G, bestc, verbose=False)
excess, deficiency = qf.morph.excess_deficiency(xi, G, B, verbose=False)

with open(args.output_basename + "-aggl-repair.txt", "w") as txt:
    txt.write("Excess / deficiency / total error: {} / {} / {}\n".format(excess, deficiency, excess + deficiency))
    txt.write("Arcs to remove or add\n\n")
    original_stdout = sys.stdout
    sys.stdout = txt
    Gp, xip = qf.morph.repair(xi, G, B, verbose=True)
    sys.stdout = original_stdout
    
qf.graphs.save(Gp, args.output_basename + "-repaired.dot", args.output_basename + "-repaired.png", colors=bestc)

# Final minimum base
ccp = qf.cc.cardon_crochemore(Gp)
Gphat = qf.graphs.minimum_base(Gp, ccp)
ccpn =  len(set(ccp.values()))
ccpnmi = qf.util.nmi(gt, ccp)
results["Final"]=(ccpn,ccpnmi)

qf.graphs.save(G, args.output_basename + "-orig-final.dot", args.output_basename + "-orig-final.png", colors=ccp)

# Compute and save difference
Gdif = qf.graphs.difference(Gp, G)
qf.graphs.save(Gdif, args.output_basename + "-orig-final-diff.dot", args.output_basename + "-orig-final-diff.png", colors=ccp, labelNodes=True, labelArcs=False)
qf.graphs.save(qf.graphs.to_simple(Gphat), args.output_basename + "-orig-final-base.dot", args.output_basename + "-orig-final-base.png", colors=ccp, labelNodes=False, labelArcs=False)

Bgt, xigt = qf.morph.qf_build(G, gt, verbose=False)
Ggtp, xigtp = qf.morph.repair(xigt, G, Bgt, verbose=False)
gtp = qf.cc.cardon_crochemore(Ggtp)
Ggtphat = qf.graphs.minimum_base(Ggtp, gtp)
Ggtdif = qf.graphs.difference(Ggtp, G)
qf.graphs.save(Ggtdif, args.output_basename + "-orig-gt-diff.dot", args.output_basename + "-orig-gt-diff.png", colors=gt, labelNodes=True, labelArcs=False)
qf.graphs.save(qf.graphs.to_simple(Ggtphat), args.output_basename + "-orig-gt-base.dot", args.output_basename + "-orig-gt-base.png", colors=gtp, labelNodes=False, labelArcs=False)

# Clusters
logging.info("Writing clustering information")
with open(args.output_basename + "-clusters.tsv", "w") as txt:
    txt.write("# Every line contains: node, ground-truth cluster (=Cardon-Crochemore, if unavailable), Cardon-Crochemore cluster, Agglomerative Clustering cluster, Final cluster\n")
    for node in G.nodes:
        txt.write("{}\t{}\t{}\t{}\t{}\n".format(node, gt[node], cc[node], bestc[node], ccp[node]))


# Comparison
logging.info("Writing cluster comparison information")
with open(args.output_basename + "-cc-vs-gt.txt", "w") as txt:
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if gt[nodes[i]] == gt[nodes[j]] and cc[nodes[i]] != cc[nodes[j]]:
                txt.write("{} and {} erroneously separated\n".format(nodes[i], nodes[j]))
            if gt[nodes[i]] != gt[nodes[j]] and cc[nodes[i]] == cc[nodes[j]]:
                txt.write("{} and {} erroneously merged\n".format(nodes[i], nodes[j]))

with open(args.output_basename + "-aggl-vs-gt.txt", "w") as txt:    
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if gt[nodes[i]] == gt[nodes[j]] and bestc[nodes[i]] != bestc[nodes[j]]:
                txt.write("{} and {} erroneously separated\n".format(nodes[i], nodes[j]))
            if gt[nodes[i]] != gt[nodes[j]] and bestc[nodes[i]] == bestc[nodes[j]]:
                txt.write("{} and {} erroneously merged\n".format(nodes[i], nodes[j]))

with open(args.output_basename + "-final-vs-gt.txt", "w") as txt:    
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if gt[nodes[i]] == gt[nodes[j]] and ccp[nodes[i]] != ccp[nodes[j]]:
                txt.write("{} and {} erroneously separated\n".format(nodes[i], nodes[j]))
            if gt[nodes[i]] != gt[nodes[j]] and ccp[nodes[i]] == ccp[nodes[j]]:
                txt.write("{} and {} erroneously merged\n".format(nodes[i], nodes[j]))


# ### All results
with open(args.output_basename + "-data.txt", "w") as txt:
    txt.write("Dataset: {}\n".format(dataset_name))
    txt.write("Ground truth: {}\n".format(args.ground))
    txt.write("Nodes: {}\nArcs: {}\n".format(G.number_of_nodes(), G.number_of_edges()))
    txt.write("Ground truth: #classes={}\n".format(gtn))
    sorted_keys = sorted(results.keys(), key=lambda x: results[x][1], reverse=True)  
    for k in sorted_keys:
        txt.write("{:40}\t{:.5}\t\t{:2}\n".format(k, results[k][1], results[k][0]))
