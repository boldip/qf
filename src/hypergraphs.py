
import hypernetx as hnx
import matplotlib as mp
import matplotlib.pyplot as plt
import networkx as nx
import qf.cc
import qf.graphs
import qf.morph
import numpy as np
import tempfile
import libsbml
import csv
from IPython.display import Image
from collections import defaultdict

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
        Saves a given hypergraph onto a given PNG file. Internal use only.
        
        Args:
            H: a `hypernetx.Hypergraph`, all of whose hyperedges contain a target propery, that contains the target node.
            png_filename: the name of the PNG file to be saved.
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
    """
        Visualizes a given hypergraph.
        
        Args:
            H: a `hypernetx.Hypergraph`, all of whose hyperedges contain a target propery, that contains the target node.
            colors: a dictionary of colors (i.e., values for each node; same value imply same color).
    """
    png_filename = tempfile.NamedTemporaryFile(suffix=".png").name
    _visualize_hg(H, png_filename, colors = colors)
    return Image(filename=png_filename)

def save_hg(H, png_filename, colors = None):
    """
        Saves a given hypergraph onto a given PNG file.
        
        Args:
            H: a `hypernetx.Hypergraph`, all of whose hyperedges contain a target propery, that contains the target node.
            png_filename: the name of the PNG file to be saved.
            colors: a dictionary of colors (i.e., values for each node; same value imply same color).
    """
    _visualize_hg(H, png_filename, colors = colors)



def hyper_cardon_crochemore(H, nodes_only=False):
    """
        Performs naive vertex refinement on the hypegraph H. 
        
        Args:
            H: the hypergraph.
            nodes_only: if set, only nodes of the hypergraph are considered.
            
        Returns:
            a dictionary whose keys are the nodes (and possibly the hyperarcs) of H, and where two keys 
            are in the same equivalence class iff they have the same value. Note that node classes and
            hyperarc classes are always disjoint.
    """
    G, dd = hg2g(H)
    cc = qf.cc.cardon_crochemore(G, starting_label=dd)
    if nodes_only:
        for x in G.nodes():
            if dd[x] == 1:
                del cc[x]
    return cc



def read_hg_from_csv(filename, max_lines = -1):
    """
        Reads and returns a hypergraph from a CSV file. The file contains one triple per line,
        where the first two elements are the sources and the last element is the target.
        
        Args:
            filename: the name of the CSV file.
            max_lines: stop after reading this number of lines (-1 for all).
            
        Returns:
            a hypergraph
    """
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


def read_hg_from_S(filename):
    """
        Reads an S-matrix file into a hypergraph. The first line of the file contains the names of the hyperarcs (the first entry should be ignored).
        The remaining lines contain as first element the name of a node, while the remaining entries are numbers:
        "1" is interpreted as the node being a source of the hyperarc; "-1" is interpreted as the node being a
        target; "0" is interpreted as the node not involved in the hyperarc; any other value is ignored.
        Hyperarcs remaining with no source or no target are ignored.
        
        Args:
            filename: the name of the file to be read.
            
        Returns:
            the hypergraph.
    """
    H = hnx.Hypergraph()
    sources = {}
    targets = {}
    with open(filename) as f:
        reader = csv.reader(f)
        hyperarcNames = next(reader)[1:]
        for a in hyperarcNames:
            sources[a] = []
            targets[a] = []
        for line in reader:
            nodeName = line[0]
            for i in range(1, len(line)):
                if float(line[i]) == +1:
                    sources[hyperarcNames[i - 1]] += [nodeName]
                elif float(line[i]) == -1:
                    targets[hyperarcNames[i - 1]] += [nodeName]
                elif float(line[i]) != 0:
                    print("Ignoring value ", line[i])
    for a in hyperarcNames:
        if len(sources[a]) == 0:
            print(a, " has no sources")
            continue
        if len(targets[a]) == 0:
            print(a, " has no targets")
            continue
        if len(targets[a]) == 1:
            add_directed_hyperedge(H, sources[a], targets[a][0], a)
        else:
            for i,t in enumerate(targets[a]):
                add_directed_hyperedge(H, sources[a], t, a + "_" + str(i))
    return H


def read_hg_from_SBML(filename):
    """
        Reads an SBML file into a hypergraph. This hypergraph contains one group of hyperarcs for every reaction,
        where reactants are the sources, and product(s) are the target(s). If there is more than one product,
        many hyperarcs are added, and their names are of the form "R_k" where "R" is the reaction id and
        "k" is the number identifying the subreaction so generated.
        
        Args:
            filename: the name of the file to be read.
            
        Returns:
            H: the hypergraph.
            sr2r: a dictionary that maps hyperarc names to reactions (i.e., reaction ids): for reactions R with just product, the 
                dictonary will contain a key equal to R, with associated value also equal to R; for
                reactions with n products, the dictionary will contain keys R_0,R_1,... and value R.
            r2ss: a dictionary that maps reactions to subsystems.
            rid2rname: a dictionary mapping reaction ids to reaction names.
            mid2mname: a dictionary mapping metabolites (i.e., species, in the SBML jargon) id to metabolites names.
    """

    H = hnx.Hypergraph()
    r2ss = {}
    sr2r = {}
    rid2rname = {}
    mid2mname = {}
    document = libsbml.readSBMLFromFile(filename)
    model = document.getModel()
    for species in model.getListOfSpecies():
        mid2mname[species.id] = species.name
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)
        reactionId = reaction.id
        rid2rname[reactionId] = reaction.name
        notes = reaction.getNotes()
        numNotes = notes.getNumChildren()
        for nn in range(numNotes):
            if notes.getChild(nn).getChild(0).toString().startswith("SUBSYSTEM:"):
                r2ss[reactionId] = notes.getChild(nn).getChild(0).toString().split(": ")[1]
        reactantsIds = [p.species for p in reaction.getListOfReactants()]
        productsIds = [p.species for p in reaction.getListOfProducts()]
        numProducts = len(productsIds)
        numReactants = len(reactantsIds)
        if numProducts > 1:
            for np, product in enumerate(productsIds):
                add_directed_hyperedge(H, reactantsIds, product, reactionId + "_" + str(np))
                sr2r[reactionId + "_" + str(np)] = reactionId
        elif numProducts == 1:
            add_directed_hyperedge(H, reactantsIds, productsIds[0], reactionId)
            sr2r[reactionId] = reactionId
        else:
            print("Ignoring reaction ", reactionId, " because it has no products")
        if reaction.reversible:
            if numReactants > 1:
                for nr, reactant in enumerate(reactantsIds):
                    add_directed_hyperedge(H, productsIds, reactant, reactionId + "_R" + str(nr))
                    sr2r[reactionId + "_R" + str(nr)] = reactionId
            elif numReactants == 1:
                sr2r[reactionId + "_R"] = reactionId
                add_directed_hyperedge(H, productsIds, reactantsIds[0], reactionId + "_R")
    return H, sr2r, r2ss, rid2rname, mid2mname

def read_hg_from_SBMLs(filenames):
    """
        Reads a (set of SBML) file(s) into a hypergraph. This hypergraph contains one group of hyperarcs for every reaction,
        where reactants are the sources, and product(s) are the target(s). If there is more than one product,
        many hyperarcs are added, and their names are of the form "R_k" where "R" is the reaction id and
        "k" is the number identifying the subreaction so generated.
        
        Args:
            filename: the name of the file to be read, or the list of name of files.
            
        Returns:
            H: the hypergraph.
            sr2r: a dictionary that maps hyperarc names to reactions (i.e., reaction ids): for reactions R with just product, the 
                dictonary will contain a key equal to R, with associated value also equal to R; for
                reactions with n products, the dictionary will contain keys R_0,R_1,... and value R.
            r2ss: a dictionary that maps reactions to subsystems.
            rid2rname: a dictionary mapping reaction ids to reaction names.
            mid2mname: a dictionary mapping metabolites (i.e., species, in the SBML jargon) id to metabolites names.
    """

    H = hnx.Hypergraph()
    r2ss = {}
    sr2r = {}
    rid2rname = {}
    mid2mname = {}
    if not isinstance(filenames, list):
        filenames = [filenames]
    for filename in filenames:
        document = libsbml.readSBMLFromFile(filename)
        model = document.getModel()
        if model is None:
            print("Ignoring file {}, couldn't get a model out of it".format(filename))
            continue
        for species in model.getListOfSpecies():
            mid2mname[species.id] = species.name
        for i in range(model.getNumReactions()):
            reaction = model.getReaction(i)
            reactionId = reaction.id
            rid2rname[reactionId] = reaction.name
            notes = reaction.getNotes()
            numNotes = notes.getNumChildren()
            for nn in range(numNotes):
                if notes.getChild(nn).getChild(0).toString().startswith("SUBSYSTEM:"):
                    r2ss[reactionId] = notes.getChild(nn).getChild(0).toString().split(": ")[1]
            reactantsIds = [p.species for p in reaction.getListOfReactants()]
            productsIds = [p.species for p in reaction.getListOfProducts()]
            numProducts = len(productsIds)
            numReactants = len(reactantsIds)
            if numProducts > 1:
                for np, product in enumerate(productsIds):
                    add_directed_hyperedge(H, reactantsIds, product, reactionId + "_" + str(np))
                    sr2r[reactionId + "_" + str(np)] = reactionId
            elif numProducts == 1:
                add_directed_hyperedge(H, reactantsIds, productsIds[0], reactionId)
                sr2r[reactionId] = reactionId
            else:
                print("Ignoring reaction ", reactionId, " because it has no products")
            if reaction.reversible:
                if numReactants > 1:
                    for nr, reactant in enumerate(reactantsIds):
                      add_directed_hyperedge(H, productsIds, reactant, reactionId + "_R" + str(nr))
                      sr2r[reactionId + "_R" + str(nr)] = reactionId
                elif numReactants == 1:
                    sr2r[reactionId + "_R"] = reactionId
                    add_directed_hyperedge(H, productsIds, reactantsIds[0], reactionId + "_R")
    return H, sr2r, r2ss, rid2rname, mid2mname


def showMapKeys(m, onlyNonTrivial = True, keyadj = None, keydesc = None, prefOnly = "", keyfamily = None):
    """
        Shows (prints) the keys of a dictionary, grouped by value. 
        
        Args:
            m: the dictionary whose keys must be shown.
            keyadj: if set, every key is adjusted (before using it) by passing it through this map.
            onlyNonTrivial: if set, only groups of at least two keys are shown.
            keydesc: if set, every key is shown accompanied with a name obtained applying this dictionary to the key.
            prefOnly: only keys starting with this prefix are considered.
            keyfamily: if set, after each group of keys a group of families is also shown, containing the families of
                the elements of the group (each family is obtained applying this dictionary to each key)
    """
    for v in set(m.values()):
        if keyadj is not None:
            eqClass = set([keyadj[k] for k,w in m.items() if w==v and k.startswith(prefOnly)])
        else:
            eqClass = set([k for k,w in m.items() if w==v and k.startswith(prefOnly)])
        if onlyNonTrivial and len(eqClass) <= 1:
            continue
        if keydesc is not None:
            out = set(["{} ({})".format(k, keydesc[k]) for k in eqClass])
        else:
            out = set(["{}".format(k) for k in eqClass])
        if keyfamily is not None and eqClass.issubset(keyfamily.keys()):
            print(out, "->", set([keyfamily[k] for k in eqClass]))
        else:
            print(out)



def classSizeDist(m, prefOnly = "", keyadj = None):
    """
        Returns a dictionary giving the distribution of the sizes of the equivalence
        classes defined by m. An equivalence class is a group of keys of m with the same value.
        The resulting dictionary will have as keys the sizes of the equivalence classes, and as
        values the number of classes of that size.
        
        Args:
            m: the map to be analyzed.
            prefOnly: only keys with this prefix are considered.
            keyadj: if specified, keys are passed through this dictionary.
            
        Return:
            a dictionary having as keys the sizes of the equivalence classes, and as
            values the number of classes of that size.
    """
    d = defaultdict(lambda: 0)
    for v in set(m.values()):
        if keyadj is not None:
            eqClass = set([keyadj[k] for k,w in m.items() if w==v and k.startswith(prefOnly)])
        else:
            eqClass = set([k for k,w in m.items() if w==v and k.startswith(prefOnly)])
        d[len(eqClass)] += 1
    return {k:v for k,v in d.items() if k > 0}


