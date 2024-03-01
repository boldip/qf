#!/usr/bin/env python
# coding: utf-8

# In[1]:


from hypergraphs import *
from kegg import *
import pandas as pd
import os
import logging

from Bio.KEGG.KGML.KGML_parser import read
import Bio.KEGG.KGML.KGML_pathway
from ast import literal_eval
from urllib.request import urlopen
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import requests
from Levenshtein import ratio
import scipy.stats
import pathlib
from qf.cc import cardon_crochemore_colored, cardon_crochemore
import random

from reportlab.lib.units import inch
import PIL.Image
from io import BytesIO
import reportlab.pdfgen.canvas
import reportlab.lib.colors


# In[2]:


organismDict = {"hsa": "Homo sapiens (human)"}
outputDirectory = "/Users/boldi/Desktop/pw/"
inputDirectory = "../../LaTeX/Data/KEGG-Pathways/Mar2024/"
dataDirectory = "../../LaTeX/Data/KEGG-Pathways"


# In[3]:


pathwaylistFilename = "KEGG-pathways.txt"  # In inputDirectory, csv: ID, type, name
superpathway = "All"                  # Name of the superpathway
organism = "hsa"
confidence = 0.05

produceGraphsWithCCcoloring = False
produceGraphsWithAnomalies = True
useCCcoloringForAnomalies = False
produceAnomaliesForNonsignificantPairs = True


# In[4]:


safeSuperpathway = makesafe(superpathway)


# In[5]:


pws = pd.read_csv(os.path.join(inputDirectory, "KEGG-pathways.txt"), 
            names=["Pathway ID", "Pathway type", "Pathway name"], dtype="string", keep_default_na=False)


# In[6]:


superpathwayDict = {
    superpathway: [(x,y) for x,y in zip(pws["Pathway ID"], pws["Pathway name"])]
}


# In[7]:


def words2set(words):
    """
        Given a space-separated list of words, returns it as a set of words
    """
    return set(words.split())


# In[8]:


from typing import NamedTuple

class Relation(NamedTuple):
    """
        Represents a relation. It is characterized by:
        - a type (e.g. "PPRel") 
        - a list of subtypes (e.g. [('activation', '-->'), ('indirect effect', '..>')])
        - a set of components (source)
        - a list of sets of components (target)
    """
    relType: str
    relSubtypes: list
    source: set
    target: list
    sourceg: Bio.KEGG.KGML.KGML_pathway.Graphics
    targetg: Bio.KEGG.KGML.KGML_pathway.Graphics
        
    @classmethod
    def fromKEGG(cls, relation, pathway):
        """
            Constructor: builds a Relation from a KEGG relation
        """
        if relation.entry2.name == "undefined":
            s = []
            for component in relation.entry2.components:
                s += [words2set(pathway.entries[component.id].name)]
            targetg = pathway.entries[component.id].graphics[0]
        else:
            s = [words2set(relation.entry2.name)]
            targetg = relation.entry2.graphics[0]
        return cls(relation.type, relation.subtypes, words2set(relation.entry1.name), s, targetg, relation.entry1.graphics[0])
        


# In[9]:


def relationsFromKEGGpathway(pathway):
    """
        Given a KEGG pathway, returns the list of relations it contains (represented as Relation).
    """
    return [Relation.fromKEGG(r, pathway) for r in pathway.relations]
    


# In[10]:


def pathwayFromFile(dataDirectory, organism, pathwayID):
    """
        Load and return a KEGG pathway, given the data directory, the organism specifying the subdirectory,
        and the pathwayID (the filename should be organism+pathwayID+"xml".
    """
    return KGML_parser.read(open(os.path.join(dataDirectory, organism, organism+pathwayID+".xml"), "r"))


# In[11]:


def relationsFromFile(dataDirectory, organism, pathwayID):
    """
        Given a pathway file (specified by a root dataDirectory, the name of the organism subdirectory, and the name
        of the file [organism+pathwayID.xml]), reads it and returns the list of its relations.
    """
    pathway = pathwayFromFile(dataDirectory, organism, pathwayID)
    return relationsFromKEGGpathway(pathway)


# In[12]:


def relationsForSuperpathway(dataDirectory, organism, superpathwayDict, superpathway):
    """
        Given a superpathDict (whose keys are superpathway names and whose values are list of
        pairs (pathwayID, pathwayName)), reads all the pathway file for a specific superpathway name
        and returns a dictionary whose keys are the pathwayID's and whose values are the list of relations.
    """
    rel = {}
    for k,v in superpathwayDict[superpathway]:
        #logging.info("Reading", k, v)
        rel[k] = relationsFromFile(dataDirectory, organism, k)
    return rel


# In[13]:


def subtypes(rel):
    """
        Given a map whose values are relations, accumulate all subtypes appearing and assign them a number.
        The result is a map from subtype string to number.
    """
    all_subtypes = set([])
    for pathwayid, relations in rel.items():
        for relation in relations:
            all_subtypes |= set([str(relation.relSubtypes)])
        
    s = sorted(list(all_subtypes))
    subtype2color = {v:k for k,v in enumerate(s)}
    color2subtype = {k:v for k,v in enumerate(s)}

    return subtype2color, color2subtype


# In[14]:


def set2can(s):
    """
        Convert a set to a string in a canonical way.
    """
    return str(sorted(list(s)))

def can2set(c):
    """
        Does the converse of set2can.
    """
    return set(literal_eval(c))


# In[15]:


def indices(rel):
    """
        Given a map whose values are lists of relations, it considers all the relations one by one, and attributes a unique id 
        to each element (i.e., set of components appearing as source or in the target of some relation) and block (the set of
        element id appearing as target of some relation).
        This function returns the dictionaries to move from/to an element or block to the corresponding id.
        Elements are string representations of sorted lists of strings.
        Blocks are string representations of sorted lists of ints.
    """
    element2id = {}
    id2element = {}
    block2id = {}
    id2block = {}

    for relations in rel.values():
        for relation in relations:
            s = set2can(relation.source)
            if s not in element2id.keys():
                element2id[s] = len(element2id)
            targetset = set([])
            for targ in relation.target:
                t = set2can(targ)
                if t not in element2id.keys():
                    element2id[t] = len(element2id)
                targetset |= set([element2id[t]])
            ts = set2can(targetset)
            if ts not in block2id.keys():
                block2id[ts] = len(block2id)

    for k,v in element2id.items():
        id2element[v] = k
    for k,v in block2id.items():
        id2block[v] = k
    return element2id, id2element, block2id, id2block


# In[16]:


def search_gene_KEGG(geneID):
    """
        Search for gene on KEGG. 
        
        Returns list of names.
    """
    url1 = "https://www.kegg.jp/entry/"+geneID
    response1 = requests.get(url1, allow_redirects=False)    
    soup = BeautifulSoup(response1.text, "html.parser")
    tds = [tds for tds in soup.find_all("td", {"class": "td11 defd"})]
    names = tds[0].getText().strip().split(", ")
    return names


# In[17]:


def get_temp_imagefilename(url):
    """Return filename of temporary file containing downloaded image.

    Create a new temporary file to hold the image file at the passed URL
    and return the filename.
    """
    img = urlopen(url).read()
    im = PIL.Image.open(BytesIO(img))
    # im.transpose(Image.FLIP_TOP_BOTTOM)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fname = f.name
    f.close()
    im.save(fname, "PNG")
    return fname


#def enhance_method(klass, method_name, replacement):
#    'replace a method with an enhanced version'
#    method = getattr(klass, method_name)
#    def enhanced(*args, **kwds): return replacement(*args, **kwds)
#    setattr(klass, method_name, enhanced)

def new_draw(self, filename, relations, elaborateArc = None):
        """Add the map elements to the drawing."""
        # Instantiate the drawing, first
        # size x_max, y_max for now - we can add margins, later
        if self.import_imagemap:
            # We're drawing directly on the image, so we set the canvas to the
            # same size as the image
            if os.path.isfile(self.pathway.image):
                imfilename = self.pathway.image
            else:
                imfilename = get_temp_imagefilename(self.pathway.image)
            im = PIL.Image.open(imfilename)
            cwidth, cheight = im.size
        else:
            # No image, so we set the canvas size to accommodate visible
            # elements
            cwidth, cheight = (self.pathway.bounds[1][0], self.pathway.bounds[1][1])
        # Instantiate canvas
        self.drawing = reportlab.pdfgen.canvas.Canvas(
            filename,
            bottomup=0,
            pagesize=(
                cwidth * (1 + 2 * self.margins[0]),
                cheight * (1 + 2 * self.margins[1]),
            ),
        )
        self.drawing.setFont(self.fontname, self.fontsize)
        # Transform the canvas to add the margins
        self.drawing.translate(
            self.margins[0] * self.pathway.bounds[1][0],
            self.margins[1] * self.pathway.bounds[1][1],
        )
        # Add the map image, if required
        if self.import_imagemap:
            self.drawing.saveState()
            self.drawing.scale(1, -1)
            self.drawing.translate(0, -cheight)
            self.drawing.drawImage(imfilename, 0, 0)
            self.drawing.restoreState()
        # Add the reactions, compounds and maps
        # Maps go on first, to be overlaid by more information.
        # By default, they're slightly transparent.
        if self.show_maps:
            self.__add_maps()
        if self.show_reaction_entries:
            KGMLCanvas._KGMLCanvas__add_reaction_entries(self)
        if self.show_orthologs:
            KGMLCanvas._KGMLCanvas__add_orthologs(self)
        if self.show_compounds:
            KGMLCanvas._KGMLCanvas__add_compounds(self)
        if self.show_genes:
            KGMLCanvas._KGMLCanvas__add_genes(self)
        # TODO: complete draw_relations code
        # if self.draw_relations:
        #    self.__add_relations()
        # Write the pathway map to PDF
        if elaborateArc is not None:
            for relation in relations:
                elaborateArc(relation, self.drawing)
        self.drawing.save()
        
#enhance_method(KGMLCanvas, 'draw',  new_draw)
setattr(KGMLCanvas, 'new_draw', new_draw)


# In[18]:


# Build all data needed for the relations of the superpathway
rel = relationsForSuperpathway(dataDirectory, organism, superpathwayDict, superpathway)
element2id, id2element, block2id, id2block = indices(rel)
subtype2color, color2subtype = subtypes(rel)
elements = list(set.union(*[can2set(k) for k in element2id.keys()]))


# In[19]:


# name2gene is a function mapping names to gene IDs (hsa:xxxx)
# e.g. KCNE3, BRGDA6 etc are all mapped to hsa:10008 (see https://www.kegg.jp/entry/hsa:10008)
# This map was produced by scraping kegg, but this has been done only once.
# After that, we just read a pkl file

name2keggFilename = os.path.join(inputDirectory, "name2kegg.pkl")

if os.path.exists(name2keggFilename):
    print("Reading gene names")
    with open(name2keggFilename, "rb") as handle:
        name2gene = pickle.load(handle)
else:
    print("Producing gene names")
    d = {}
    nelements = len(elements)
    for count,element in enumerate(elements):
        if count % 10 == 0:
            print(f"{count}/{nelements}")
        if element.startswith("hsa"):
            names = search_gene_KEGG(element)
            for name in names:
                d[name] = element
    with open(name2keggFilename, "wb") as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    name2gene = d


# In[20]:


# Read patient data
print("Reading patient data")
genes = pd.read_csv(os.path.join(inputDirectory, "genes.csv"), dtype={"name": str})
patients = genes["name"].values.tolist()
classification = pd.read_csv(os.path.join(inputDirectory, "classification.csv"), dtype={"name": str})

patient2class = {patient: classification.loc[classification["name"]==patient]["classification"].values.flatten().tolist()[0] for patient in patients}
numHealthy = sum([x == 'healthy' for x in patient2class.values()])
numDiseased = sum([x == 'diseased' for x in patient2class.values()])


# In[21]:


# Dividing genes into two frames (one for healthy individuals and one for diseased individuals)
genesHealthy = genes.loc[genes["name"].apply(lambda x: patient2class[x])=="healthy"]
genesDiseased = genes.loc[genes["name"].apply(lambda x: patient2class[x])=="diseased"]


# In[22]:


# Building the graph of all relations contained in rel values
# Nodes are IDs, and node x correspond to the set of genes (and/or compound) id2element[x]
print("Building the relation graph")
G = nx.DiGraph() 
for pathwayID, relations in rel.items():
    for relation in relations:
        source = element2id[set2can(relation.source)]
        for targ in relation.target:
            target = element2id[set2can(targ)]
            newType = str(relation.relSubtypes)
            color = subtype2color[newType]
            pl = []
            if G.has_edge(source, target):
                d = G.get_edge_data(source, target)
                if d["color"] != color:
                    oldType = color2subtype[d["color"]]
                    print("- Edge ({},{}) of type {} for {} appears as {} for {}. ".format(
                        source, target, color2subtype[d["color"]], d["pathwayIDs"], 
                        str(relation.relSubtypes), pathwayID
                    ), end="")
                    if oldType[1:-1] in newType[1:-1]:
                        print("Overwriting with type {}".format(newType))
                    elif newType[1:-1] in oldType[1:-1]:
                        print("Overwriting with type {}".format(oldType))
                        color = subtype2color[oldType]
                    else:
                        print("\n****Conflict! Rewriting as {}".format(newType))
                pl = d["pathwayIDs"]
            G.add_edge(source, target, color=color, pathwayIDs=pl + [pathwayID])
            

# Computing the minimum fibres
cc=cardon_crochemore_colored(G)

colors = list(matplotlib.colors.CSS4_COLORS.values())
usedColors = 0

cc2fibresize = {}
cc2fibre = {}
ccnontrivial = {}
nontrivialclass2col = {}

fibres = []
node2fibre = {}
for v in set(cc.values()):
    fibre = [x for x in G.nodes() if cc[x] == v]
    fibres += [fibre]
    for x in fibre:
        node2fibre[x] = fibre
    cc2fibre[v] = fibre
    cc2fibresize[v] = len(fibre)
    if len(fibre) > 0:
        ccnontrivial[v] = cc[v]
        nontrivialclass2col[v] = colors[usedColors]
        usedColors = (usedColors + 1 ) % len(colors)
        if usedColors == 0:
            print("Wrap: more than", len(colors), "colors needed")


# In[23]:


# Build node2columns, mapping each node of G to the list of genes corresponding to it
# for which we know a value (in the genes dataframe), sorted
#

print("Mapping genes to columns")
namedGenes = set(name2gene.values())     
columns = set(genes.columns.values.tolist()[2:])
node2columns = {}
for node in G.nodes():
    nodeGenes = can2set(id2element[node])
    s = []
    for gene in nodeGenes & namedGenes:
        for k,v in name2gene.items():
            if v == gene:
                if k in columns:
                    s += [k]
    node2columns[node]=sorted(s)


# In[24]:


if produceGraphsWithCCcoloring:
    s = ""
    for pair in superpathwayDict[superpathway]:
        pathwayID = pair[0]
        pathway = pathwayFromFile(dataDirectory, organism, pathwayID)
        canvas = KGMLCanvas(pathway)
        for k in pathway.entries:
            name = pathway.entries[k].name
            canonicalName = set2can(words2set(name))
            if canonicalName in element2id.keys():
                node = element2id[canonicalName]
                klass = cc[node]
                if klass in nontrivialclass2col:
                    pathway.entries[k].graphics[0].bgcolor = nontrivialclass2col[klass]
                    pathway.entries[k].graphics[0].fgcolor = nontrivialclass2col[klass]
                else:
                    pathway.entries[k].graphics[0].bgcolor = "#FFFFFF"
                    pathway.enries[k].graphics[0].fgcolor = "#FFFFFF"
                s += pathwayID + "\t" + str(node) + "\t" + str(klass) +"\t"+pathway.entries[k].graphics[0].name+"\t"+canonicalName + "\n"
        canvas.import_imagemap = True
        pdfName = organism + pathwayID + ".pdf"
        pathlib.Path(os.path.join(outputDirectory, organism, safeSuperpathway)).mkdir(parents=True, exist_ok=True)
        canvas.draw(os.path.join(outputDirectory, organism, safeSuperpathway, pdfName))
    with open(os.path.join(outputDirectory, organism, safeSuperpathway, "all.tsv"), "wt") as file:
        file.write(s)


# In[25]:


def columns2nodes(columns, node2columns, ignoreEmpty = True):
    """
        Returns a map from columns to the nodes where the column appears.
    """
    res = {} 
    for c in columns:
        s = set([])
        for no, co in node2columns.items():
            if c in co:
                s |= set([no])
        if len(s) > 0 or not ignoreEmpty:
            res[c] = s
    return res

c2n = columns2nodes(columns, node2columns)
usableColumns = list(c2n.keys())


# In[26]:


def corrForPair(genes, set1, set2, correlationFunction = lambda x: scipy.stats.kendalltau(x).statistic):
    """
        Given a dataframe genes (i.e., a set of patients), and two sets of column names (set1, set2), 
        compute the correlation function
        for every pair of columns in set1 and set2 and return the average statistic and its std.
    """
    taus = []
    for gene1 in set1:
        for gene2 in set2:
            expr1 = genes.loc[:,gene1].values.tolist()
            expr2 = genes.loc[:,gene2].values.tolist()
            taus += [scipy.stats.kendalltau(expr1, expr2).statistic]
    return pd.DataFrame(taus).mean().values[0], pd.DataFrame(taus).std().values[0]


# In[27]:


def sampleDeltaDist(genes, set1, set2, numberOfRows1, numberOfSamples = 500, correlationFunction = lambda x: scipy.stats.kendalltau(x).statistic):
    """
        Repeatedly (for numberOfSamples times) do the following:
        - select numberOfRows1 rows of genes, and then compute tauForPair on those rows for set1 and set2
        - do the same for remaining rows
        - take the difference between the two resulting set of taus.
        Return the mean and std of differences.
    """
    diffs = []
    
    for i in range(numberOfSamples):
        genes1 = random.sample(range(len(genes)), numberOfRows1)
        genes2 = list(set(range(len(genes)))-set(genes1))
        mt1, mstd1 = corrForPair(genes.iloc[genes1], set1, set2, correlationFunction)
        mt2, mstd2 = corrForPair(genes.iloc[genes2], set1, set2, correlationFunction)
        diffs += [mt1-mt2]
    return diffs


def deltaForPair(genes1, genes2, set1, set2, correlationFunction = lambda x: scipy.stats.kendalltau(x).statistic):
    """
        Compute corrForPair for set1 and set2 over the set of rows genes1 and genes2, and return
        the difference of taus.
    """
    mt1, mstd1 = corrForPair(genes1, set1, set2, correlationFunction)
    mt2, mstd2 = corrForPair(genes2, set1, set2, correlationFunction)
    return mt1-mt2


# In[28]:


correlationFunction = lambda x:  scipy.stats.spearmanr(x).statistic
correlationFunctionName = "spearmanrho"
relAnomalies = os.path.join(inputDirectory, "anomalies-" + correlationFunctionName + ".csv")

if os.path.exists(relAnomalies):
    print("Reading anomalies")
    relcorrel = pd.read_csv(relAnomalies)
else:
    print("Producing anomalies")
    relcorrel = pd.DataFrame(columns=[
        "Source gene(s)",
        "Target gene(s)",
        "Pathway ID",
        "Relation type",
        "Correlation for healthy",
        "Correlation for diseased",
        "Correlation difference (H-D)",
        "Expected correlation difference",
        "P-value",
        "Comment"
        ])
    numberOfEdges = G.number_of_edges()
    doneEdges = 0
    for s,t,d in G.edges(data=True):
        if doneEdges % 10 == 0:
            print("{:.2f}% done".format(100 * doneEdges / numberOfEdges))
        doneEdges += 1
        subtypeName = [k for k,v in subtype2color.items() if d["color"]==v][0]
        sg = node2columns[s]
        tg = node2columns[t]
        if len(sg) == 0 or len(tg) == 0:
            continue
        avgh, stdh = corrForPair(genesHealthy, sg, tg, correlationFunction=correlationFunction)
        avgd, stdd = corrForPair(genesDiseased, sg, tg, correlationFunction=correlationFunction)
        taudiff = avgh - avgd
        ttn = sampleDeltaDist(genes, sg, tg, numHealthy, correlationFunction=correlationFunction)
        tt = [x for x in ttn if ~np.isnan(x)]
        ttmean = pd.DataFrame(tt).mean().values[0]
        ecdft = scipy.stats.ecdf(tt)  # Empirical CDF
        if taudiff < 0:
            pvalue = ecdft.cdf.evaluate(taudiff)
            comment = "More correlated than for healthy individuals"
        else:
            pvalue = 1 - ecdft.cdf.evaluate(taudiff)    
            comment = "Less correlated than for healthy individuals"
        if produceAnomaliesForNonsignificantPairs or produpvalue <= confidence:
            relcorrel.loc[len(relcorrel)] = [
                str(sg),
                str(tg),
                d["pathwayIDs"],
                subtypeName,
                avgh, 
                avgd,
                taudiff,
                ttmean,
                pvalue,
                comment
            ]
    relcorrel = relcorrel.sort_values(by=["P-value"], ascending=True)
    relcorrel.to_csv(relAnomalies)


# In[ ]:


print("Anomalies by type")
statisticallySignificant = relcorrel.loc[
    (relcorrel["P-value"]<confidence)   # Statistically significant
]

failingActivation = relcorrel.loc[
    (relcorrel["P-value"]<confidence) &   # Statistically significant
    (relcorrel["Relation type"].str.contains("activation")) & # Activation  
    (relcorrel["Correlation difference (H-D)"]>0) # Less than healthy
]

failingInhibition = relcorrel.loc[
    (relcorrel["P-value"]<confidence) &   # Statistically significant
    (relcorrel["Relation type"].str.contains("inhibition")) & # Inhibition
    (relcorrel["Correlation difference (H-D)"]<0) # More than healthy
]

excessiveActivation = relcorrel.loc[
    (relcorrel["P-value"]<confidence) &   # Statistically significant
    (relcorrel["Relation type"].str.contains("activation")) & # Activation  
    (relcorrel["Correlation difference (H-D)"]<0) # More than healthy
]

excessiveInhibition = relcorrel.loc[
    (relcorrel["P-value"]<confidence) &   # Statistically significant
    (relcorrel["Relation type"].str.contains("inhibition")) & # Inhibition
    (relcorrel["Correlation difference (H-D)"]>0) # Less than healthy
]

otherTypesOfRelations = relcorrel.loc[
    (relcorrel["P-value"]<confidence) &   # Statistically significant
    ~(relcorrel["Relation type"].str.contains("activation")) & # No activation  
    ~(relcorrel["Relation type"].str.contains("inhibition"))   # No inhibition 
]

print("P-value < {:.2f}: ".format(100 * confidence / 100), len(statisticallySignificant))
print("Failing activations (green):", len(failingActivation))
print("Failing inhibitions (blue):", len(failingInhibition))
print("Excessive activations (red):", len(excessiveActivation))
print("Excessive inhibitions (orange):", len(excessiveInhibition))
print("Other types of relations (yellow):", len(otherTypesOfRelations))
print("Total:", len(failingActivation)+len(failingInhibition)+len(excessiveActivation)+len(excessiveInhibition)+len(otherTypesOfRelations))


# In[55]:





# In[ ]:


def elaborateArc(relation, sdrawing):
    source= str(node2columns[element2id[set2can(relation.source)]])
    for tg in relation.target:
        target = str(node2columns[element2id[set2can(tg)]])
        r = relcorrel.loc[(relcorrel["Source gene(s)"] == source) & (relcorrel["Target gene(s)"] == target)]
        if len(r) == 0:
            continue
        if len(r) > 1:
            pass
            #print(f"({source}, {target}) appearing more than once in relcorrel")
            #print(list(r.to_dict()["Correlation difference (H-D)"].values()))
        rd = r.to_dict()
        pvalue = list(rd["P-value"].values())[0]
        corrdiff = list(rd["Correlation difference (H-D)"].values())[0]
        relationType = list(rd["Relation type"].values())[0]
        if pvalue < confidence:        
            if "activation" in relationType:
                if corrdiff > 0:
                    col = reportlab.lib.colors.green
                else:
                    col = reportlab.lib.colors.red
            elif "inhibition" in relationType:
                if corrdiff > 0:
                    col = reportlab.lib.colors.orange
                else:
                    col = reportlab.lib.colors.blue
            else:
                col = reportlab.lib.colors.yellow
            sdrawing.setLineWidth(3*math.exp(abs(corrdiff)*3))
            sdrawing.setStrokeColor(col)
            sdrawing.line(relation.sourceg.x, relation.sourceg.y, relation.targetg.x, relation.targetg.y)  


# In[ ]:


if produceGraphsWithAnomalies:
    print("Producing anomaly PDFs")
    s = ""
    for pair in superpathwayDict[superpathway]:
        pathwayID = pair[0]
        pathway = pathwayFromFile(dataDirectory, organism, pathwayID)

        relations = []
        for relation in rel[pathwayID]:
            relations += [relation]

        canvas = KGMLCanvas(pathway)

        if useCCcoloringForAnomalies:
            for k in pathway.entries:
                name = pathway.entries[k].name
                canonicalName = set2can(words2set(name))
                if canonicalName in element2id.keys():
                    node = element2id[canonicalName]
                    klass = cc[node]
                    if klass in nontrivialclass2col:
                        pathway.entries[k].graphics[0].bgcolor = nontrivialclass2col[klass]
                        pathway.entries[k].graphics[0].fgcolor = nontrivialclass2col[klass]
                    else:
                        pathway.entries[k].graphics[0].bgcolor = "#FFFFFF"
                        pathway.entries[k].graphics[0].fgcolor = "#FFFFFF"
                    s += pathwayID + "\t" + str(node) + "\t" + str(klass) +"\t"+pathway.entries[k].graphics[0].name+"\t"+canonicalName + "\n"

        canvas.import_imagemap = True
        pdfName = organism + pathwayID + ".pdf"
        pathlib.Path(os.path.join(outputDirectory, organism, safeSuperpathway + "-anomalies")).mkdir(parents=True, exist_ok=True)
        canvas.new_draw(os.path.join(outputDirectory, organism, safeSuperpathway+ "-anomalies", pdfName), relations, elaborateArc)


# In[ ]:


print("Computing anomalies by pathway")
pathwayAnomalies = pd.DataFrame(columns=[
        "Pathway ID",
        "Pathway",
        "Average square difference",
        "Count",
        "Relative count"
        ])
for pair in superpathwayDict[superpathway]:
    pathwayID = pair[0]
    pathwayName= pair[1]
    ssqd = 0
    count = 0
    countSig = 0
    for relation in rel[pathwayID]:
        source = str(node2columns[element2id[set2can(relation.source)]])
        for targ in relation.target:
            target = str(node2columns[element2id[set2can(targ)]])
            tt = relcorrel.loc[(relcorrel["Source gene(s)"] == source) & (relcorrel["Target gene(s)"] == target)]
            if len(tt) == 1 and tt["P-value"].to_list()[0] <= confidence:
                cd = tt["Correlation difference (H-D)"].to_list()[0]
                ssqd += cd * cd
                countSig += 1
            count += 1
    if count > 0:
        pathwayAnomalies.loc[len(pathwayAnomalies)]=[pathwayID, pathwayName, ssqd, countSig, countSig / count]


# In[ ]:


print("Sorting pathways by number of anomalies and writing it out as a csv")
relPathwayAnomalies = os.path.join(inputDirectory, "pathway-anomalies-" + correlationFunctionName + ".csv")
pathwayAnomalies = pathwayAnomalies.sort_values(by=["Count"], ascending=False)
pathwayAnomalies.to_csv(relPathwayAnomalies)


# In[ ]:


print("Extracting genes involved in top-k anomalous pathways and appearing as columns")
nTopAnomPathway = 7
topAnomPathway = pathwayAnomalies.head(nTopAnomPathway)["Pathway ID"].to_list()
anomPathwayGenes = set([])
for pathwayId in topAnomPathway:
    for relation in rel[pathwayId]:
        anomPathwayGenes |= relation.source 
        for t in relation.target:
            anomPathwayGenes |= t
anomPathwayGeneNames = [name for name,gene in name2gene.items() if gene in anomPathwayGenes and name in columns]
print(len(anomPathwayGeneNames), "genes involved")


# In[ ]:


print("Computing correlations for anomalous-pathway genes in the two populations")
healthyMatrix = genesHealthy[anomPathwayGeneNames].to_numpy() #Rows are healthy patients, columns are genes appearing in anomalous pathways
diseasedMatrix = genesDiseased[anomPathwayGeneNames].to_numpy() #Rows are diseased patients, columns are genes appearing in anomalous pathways

healthyCorr = np.corrcoef(healthyMatrix, rowvar=False) #Correlation between anomalous genes in healthy individuals
diseasedCorr = np.corrcoef(diseasedMatrix, rowvar=False) #Correlation between anomalous genes in diseased individuals


# In[154]:


import sklearn.cluster
bh = sklearn.cluster.k_means(np.square(healthyCorr), n_clusters=9)[1]
bd = sklearn.cluster.k_means(np.square(diseasedCorr), n_clusters=9)[1]


# In[167]:


import sklearn.cluster
import operator

gene2fr = defaultdict(lambda: 0)

for k in range(10, 11):
    bh = sklearn.cluster.k_means(np.square(healthyCorr), n_clusters=k)[1]
    bd = sklearn.cluster.k_means(np.square(diseasedCorr), n_clusters=k)[1]

    t = 0
    tc = 0
    for i1,g1 in enumerate(anomPathwayGeneNames):
        for i2,g2 in enumerate(anomPathwayGeneNames):
            if i1<i2:
                if (bh[i1]==bh[i2]) != (bd[i1]==bd[i2]):
                    t += 1
                    #print(t, g1, g2)
                    gene2fr[g1] += 1
                    gene2fr[g2] += 1
                else:
                    tc += 1
sorted_dict = dict(sorted(gene2fr.items(), key=operator.itemgetter(1)))
print(sorted_dict.values())


# In[ ]:





# In[132]:


dbhcs, dbhlabels = dbscan(np.square(healthyCorr), metric="precomputed", eps=0.8)
len(set(dbhlabels))


# In[128]:


genesHealthy[names].to_numpy()


# In[58]:


columns


# In[ ]:




