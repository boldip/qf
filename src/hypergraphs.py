import hypernetx as hnx
import matplotlib as mp
import matplotlib.pyplot as plt
import networkx as nx
import qf.cc
import qf.graphs
import qf.morph
import numpy as np
import tempfile
import csv

def add_directed_hyperedge(H, sources, target, name=None):
    """
    Adds a directed hyperarc to the hypergraph H. The corresponding edge has all sources (a list of nodes)
    and target as endpoints. The special property 'target' is associated with the edge, whose value is the target.
    
    Args:
        H: a `hypernetx.Hypergraph`.
        sources: a list containing the source nodes.
        target: the target node.
        name: name of the hyperarc.
    """
    if name is None:
        name = H.number_of_edges()
    H.add_edge(hnx.Entity(name, sources + [target], props={'target': target}))

def add_directed_hyperedges(H, stpairs):
    """
    Adds a list directed hyperarcs, each specified as a pair (sources, target[, name]).
    
    Args:
        H: a `hypernetx.Hypergraph`.
        stpairs: a list containing the arcs to be added, each specified by a pair or triple (the first element
        being a list of sources and the target being the target). The third element, if present, is the name
        of the hyperarc.
    """
    for st in stpairs:
        if len(st) == 2:
            add_directed_hyperedge(H, st[0], st[1])
        else:
            add_directed_hyperedge(H, st[0], st[1], name=st[2])


def hg2g(H):
    """
    Returns the RB-graph representation of a hypergraph.
    
    Args:
        H: a `hypernetx.Hypergraph`, all of whose hyperedges contain a target propery, that contains the target node.
        
    Returns:
        - the graph.
        - a dictionary mapping the nodes of the returned graph to 0 (blue nodes, i.e., nodes of H) or to 1 (red nodes, i.e., hyperarcs of H).
    """
    G = nx.MultiDiGraph()
    G.add_nodes_from([x.uid for x in H.nodes()])
    G.add_nodes_from([x.uid for x in H.edges()])
    dd1 = {x.uid: 0 for x in H.nodes()} # Blue nodes
    dd2 = {x.uid: 1 for x in H.edges()} # Red nodes
    dd = {**dd1, **dd2}
    for h in H.edges():
        target = h.props['target']
        for i,e in enumerate(h.elements):
            if e != target:
                G.add_edge(e, h.uid, label='(%s,%d,%s)' % (e, i, h.uid))
            else:
                G.add_edge(h.uid, e, label='(%s,%s)' % (h.uid, e))
    return G, dd


posDict = {}

def _visualize_hg(H, png_filename, colors = None):
    """
        Visualizes a given hypergraph.
        
        Args:
            H: a `hypernetx.Hypergraph`, all of whose hyperedges contain a target propery, that contains the target node.
            colors: a dictionary of colors (i.e., values for each node; same value imply same color).
    """
    if colors is None:
        col1 = {v.uid: 0 for v in H.nodes()} 
        col2 = {h.uid: 0 for h in H.edges()}
        colors = {**col1, **col2}
    ncols = max(set(colors.values()))+1 
    cols = np.vstack([[0,0,0,1],plt.cm.tab10(np.arange(40))]) # 0 is black
    plt.ioff()
    fig, ax = plt.subplots()
    if H in posDict:
        posnodes = posDict[H]
    else:
        posnodes = hnx.drawing.rubber_band.layout_node_link(H)
        posDict[H] = posnodes
    hnx.drawing.draw(H, 
                 pos = posnodes, ax=ax,
                 with_edge_labels=False,
                 edges_kwargs={
                     'edgecolors': [cols[colors[h.uid]] for h in H.edges()]
                 },
                 nodes_kwargs={
                     'facecolors': [cols[colors[v.uid]] for v in H.nodes()]
                 }
                )
    for h in H.edges():
        target = h.props['target']
        xtarget = posnodes[target][0]
        ytarget = posnodes[target][1]
        for e in h.elements:
            if e != target:
                xsource = posnodes[e][0]
                ysource = posnodes[e][1]
                dx = xtarget - xsource
                dy = ytarget - ysource
                ax.arrow(xsource, ysource, dx, dy, head_width=.03, head_length=.03, 
                         length_includes_head=True, color=cols[colors[h.uid]])
    plt.savefig(png_filename, format="PNG")

def visualize_hg(H, colors = None):
    png_filename = tempfile.NamedTemporaryFile(suffix=".png").name
    _visualize_hg(H, png_filename, colors = colors)
    return Image(filename=png_filename)

def save_hg(H, png_filename, colors = None):
    _visualize_hg(H, png_filename, colors = colors)

def hyper_cardon_crochemore(H, nodes_only=False):
    G, dd = hg2g(H)
    cc = qf.cc.cardon_crochemore(G, starting_label=dd)
    if nodes_only:
        for x in G.nodes():
            if dd[x] == 1:
                del cc[x]
    return cc



def read_hg_from_csv(filename, max_lines = -1):
    H = hnx.Hypergraph()
    n = 0
    with open(filename) as f:
        reader = csv.reader(f)
        for triple in reader:
            if max_lines >= 0 and n >= max_lines:
                break        
            add_directed_hyperedge(H, [triple[0], triple[1]], triple[2])
            n += 1
    return H



H = read_hg_from_csv("CSO.3.3.csv")
c = hyper_cardon_crochemore(H, nodes_only=True)
for v in set(c.values()):
    print([k for k in c.keys() if c[k]==v])
    print

