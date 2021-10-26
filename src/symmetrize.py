#!/usr/bin/env python

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
import qf.qastar
import qf.qzss
import qf.util

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
argparser.add_argument("--minutes", type=int, default=60,
                       help="Maximum number of minutes for the computation of the UTD matrix")
args = argparser.parse_args() 
    
## Read file

if args.graph_filename.startswith(":"):
    logging.info("Creating synthetic graph")
    if args.ground is not None:
        argparser.error("Cannot specify ground truth for synthetic datasets")
    spl = args.graph_filename.split(":")
    if len(spl) < 3 or len(spl) > 4:
        argparser.error("Wrong filename specification {}".format(args.graph_filename))
    try:
        n = int(spl[1])
        if len(spl) == 3:
            noise = int(spl[2])
        else:
            noise = -1
            add_noise = int(spl[2])
            del_noise = int(spl[3])
    except ValueError:
        argparser.error("Wrong synthetic dataset specification {}".format(args.graph_filename))

    dataset_name = "Synthetic"
    Gideal = qf.zssexp.get_fibration_rich_graph(n)
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
        coords = qf.util.read_coordinates(args.coord, skipHeader=args.coord_skip_header, separator=args.coord_separator)    
    G = qf.util.read_graph(args.graph_filename, skipHeader=args.graph_skip_header, separator=args.graph_separator, dense=args.graph_is_dense, coordinates=coords)
    if args.ground is None:
        raise Exception("No ground truth available for real graph")
    else:
        gt = qf.util.read_label(args.ground, skipHeader=args.ground_skip_header, separator=args.ground_separator)
    gtn = len(set(gt.values()))

depth = args.depth
results = {}

# Compute Katz if needed
order_label = None
if args.katz:
    qf.qzss.katz_preorder(G, "katz")
    order_label = "katz"

# Build nodes and indices
n=G.number_of_nodes()
nodes=list(G.nodes)
indices = {}
for i in range(len(nodes)):
    indices[nodes[i]] = i


#Compute Cardon-Crochemore
logging.info("Running Cardon-Crochemore")
cc = qf.cc.cardon_crochemore(G)
ccn =  len(set(cc.values()))
ccnmi = qf.util.nmi(gt, cc)
results["Cardon-Crochemore"]=(ccn,ccnmi)

#Compute depth-limited Cardon-Crochemore
logging.info("Running depth-limited Cardon-Crochemore")
ccdl = qf.cc.cardon_crochemore(G, max_step=depth)
ccdln = len(set(ccdl.values()))

#Compute agglomerative clustering on the pure ZSS matrix
# ZSS matrix
logging.info("Running Agglomerative UED")
linkage_type = "single"
logging.info("Computing fallback OED matrix")
M, nodes, indices = qf.qzss.cached_zss_dist_matrix(G, depth)        
nM = M/sum(sum(M))
Mzss = M
logging.info("Computing UED matrix; may take long (up to {} minutes)".format(args.minutes))
M, nodes, indices = qf.qastar.qastar_dist_matrix(G, depth, Msubs=Mzss, max_milliseconds=1000*60*args.minutes, zero=ccdl)       
nM = M/sum(sum(M))
logging.info("Clustering")
c, _M, nodes, indices = qf.qastar.agclust_optcl(G, depth, min(4, n), ccdln, nM, nodes, indices, linkage_type=linkage_type)
bestc = qf.qastar.agclust2dict(c, _M, nodes, indices)
bestcn = len(set(bestc.values()))
bestcnmi = qf.util.nmi(gt, bestc)
description="Agglomerative UED (linkage={})".format(linkage_type)
results[description]=(bestcn,bestcnmi)
# Completion
logging.info("Building quasi-fibration and repairing")
B, xi = qf.morph.qf_build(G, bestc, verbose=False)
excess, deficiency = qf.morph.excess_deficiency(xi, G, B, verbose=False)
logging.info("Excess / deficiency / total error: {} / {} / {}".format(excess, deficiency, excess + deficiency))
logging.info("Repairing graph")
Gp, xip = qf.morph.repair(xi, G, B, verbose=True)

# Final minimum base
ccp = qf.cc.cardon_crochemore(Gp)
Gphat = qf.graphs.minimum_base(Gp, ccp)
ccpn =  len(set(ccp.values()))
ccpnmi = qf.util.nmi(gt, ccp)
results["Final"]=(ccpn,ccpnmi)

# Save graph files
logging.info("Writing dot/png files")
qf.graphs.save(G, args.output_basename + "-orig-gt.dot", args.output_basename + "-orig-gt.png", colors=gt)
qf.graphs.save(G, args.output_basename + "-orig-cc.dot", args.output_basename + "-orig-cc.png", colors=cc)
qf.graphs.save(G, args.output_basename + "-orig-aggl.dot", args.output_basename + "-orig-aggl.png", colors=bestc)
qf.graphs.save(G, args.output_basename + "-orig-reduced.dot", args.output_basename + "-orig-reduced.png", colors=ccp)
qf.graphs.save(Gp, args.output_basename + "-repaired.dot", args.output_basename + "-repaired.png", colors=ccp)

# Compute and save difference
Gdif = qf.graphs.difference(Gp, G)
qf.graphs.save(Gdif, args.output_basename + "-diff.dot", args.output_basename + "-diff.png", colors=ccp, labelNodes=True, labelArcs=False)
qf.graphs.save(qf.graphs.to_simple(Gphat), args.output_basename + "-base.dot", args.output_basename + "-base.png", colors=ccp, labelNodes=False, labelArcs=False)

# Clusters
logging.info("Writing clustering information")
with open(args.output_basename + "-clusters.tsv", "w") as txt:
    txt.write("# Every line contains: node, ground-truth cluster (=Cardon-Crochemore, if unavailable), Cardon-Crochemore cluster, Agglomerative Clustering cluster, Reduced aggl. cluster\n")
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

with open(args.output_basename + "-reduced-vs-gt.txt", "w") as txt:    
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

with open(args.output_basename + "-README.txt", "w") as txt:
    txt.write("TEXT FILES\n")
    txt.write(" - {}-README.txt: this file\n".format(args.output_basename))
    txt.write(" - {}-data.txt: general report\n".format(args.output_basename))
    txt.write(" - {}-cc-vs-gt.txt: symmetric difference between the arcs of Cardon-Crochemore and those required by the Ground truth\n".format(args.output_basename))
    txt.write(" - {}-reduced-vs-gt.txt: symmetric difference between the arcs of the reduced aggl. UED and those required by the Ground truth\n".format(args.output_basename))
    txt.write(" - {}-clusters.tsv: clusters according to gt/cc/aggl/reduced\n".format(args.output_basename))    
    txt.write("\n")
    txt.write("GRAPHS (in .dot and .png format)\n")
    txt.write(" - {}-orig-<XXX>.<EXT>: original graph, with different colours depending on <XXX> (<EXT> is dot or png)\n".format(args.output_basename))
    txt.write("                  gt: Ground truth clustering\n".format(args.output_basename))
    txt.write("                  cc: Cardon-Crochemore clustering\n".format(args.output_basename))
    txt.write("                  aggl: Agglomerative UED\n".format(args.output_basename))
    txt.write("                  reduced: Reduced aggl. UED\n".format(args.output_basename))
    txt.write(" - {}-repaired.<EXT>: repaired graph (reduced aggl. UED clustering)\n".format(args.output_basename))
    txt.write(" - {}-diff.<EXT>: difference between original and repaired graph (reduced aggl. UED clustering)\n".format(args.output_basename))
    txt.write(" - {}-base.<EXT>: minimum base of repaired graph, as a simple graph (reduced aggl. UED clustering)\n".format(args.output_basename))
    