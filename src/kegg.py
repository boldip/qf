from hypergraphs import *
import re
import os
import pickle
import math

from Bio import SeqIO
from Bio.KEGG.REST import *
from Bio.KEGG.KGML import KGML_parser
from Bio.Graphics.KGML_vis import KGMLCanvas
from Bio.Graphics.ColorSpiral import ColorSpiral

from IPython.display import Image, HTML
import numpy as np
import matplotlib

def makesafe(x):
    return re.sub(r'[\W_]', '', x)

def getColorFor(x):
    if not hasattr(getColorFor, "cols"):
        getColorFor.MAX_COLORS = 400
        getColorFor.rgbCols = plt.cm.get_cmap("viridis", getColorFor.MAX_COLORS).colors
        getColorFor.cols = [matplotlib.colors.rgb2hex(c) for c in getColorFor.rgbCols]
    if type(x) == list or type(x) == set:
        x = hash(sorted(x)[0]) % len(getColorFor.cols)
    return getColorFor.cols[x]


def writeKEGGHtml(m, dirname, basename, title, onlyNonTrivial = True, keyadj = None, 
                  prefOnly = "", cutPref="", addHtmlPrefix="https://www.kegg.jp/entry/", maxPerRow = 3):
    """
    """
    pageCounter = 0
    npages = len(set(m.values()))
    for v in set(m.values()):
        if keyadj is not None:
            eqClass = set([keyadj[k] for k,w in m.items() if w==v and k.startswith(prefOnly)])
        else:
            eqClass = set([k for k,w in m.items() if w==v and k.startswith(prefOnly)])
        if onlyNonTrivial and len(eqClass) <= 1:
            continue
        out = set(["{}".format(k) for k in eqClass])
        outn = len(out)
        rows = math.ceil(outn/maxPerRow)
        fileName = "{}-cont-{}.html".format(basename, pageCounter)
        prevName = "{}-cont-{}.html".format(basename, pageCounter - 1)
        nextName = "{}-cont-{}.html".format(basename, pageCounter + 1)
        headerName = "{}-header-{}.html".format(basename, pageCounter)
        footerName = "{}-footer-{}.html".format(basename, pageCounter)
        pageCounter += 1
        os.makedirs(os.path.dirname(os.path.join(dirname, fileName)), exist_ok=True)
        with open(os.path.join(dirname, fileName), "w") as file:
            file.write("<!DOCTYPE html>\n<html>\n\t<frameset rows=\"10%," +
                ",".join(["*"] * rows) +
                ",10%\">\n")
            file.write("\n\t\t<frame src=\"" + headerName + "\">\n")
            for i,element in enumerate(out):
                if i % maxPerRow == 0:
                    file.write("\t\t<frameset cols=\"" +
                        ",".join(["*"] * maxPerRow) +
                        "\">\n")
                file.write("\t\t\t<frame src=\"{}{}\">\n".format(addHtmlPrefix, element[len(cutPref):]))
                if (i + 1) % maxPerRow == 0 or i == outn - 1:
                    file.write("\t\t</frameset>\n")
            file.write("\n\t\t<frame src=\"" + footerName + "\">\n")
            file.write("\n\t</frameset>\n</html>")
        with open(os.path.join(dirname, headerName), "w") as file:
            file.write("<!DOCTYPE html>\n<html>\n\t<h1>{} #{}</h1>\n</html>".format(title, v))
        with open(os.path.join(dirname, footerName), "w") as file:
            if pageCounter == 1:
                file.write("<!DOCTYPE html>\n<html>\n" +
                       "\t<table border=\"1\" style=\"margin-left: auto; margin-right: auto;\">\n" +
                       "\t\t<td style=\"padding: 10px;\"><a href=\"" + nextName + "\" target=\"_top\">Next</a></td>\n" +
                       "\t</table>\n" +
                       "</html>")
            elif pageCounter == npages:
                file.write("<!DOCTYPE html>\n<html>\n" +
                       "\t<table border=\"1\" style=\"margin-left: auto; margin-right: auto;\">\n" +
                       "\t\t<td style=\"padding: 10px;\"><a href=\"" + prevName + "\" target=\"_top\">Previous</a></td>\n" +
                       "\t</table>\n" +
                       "</html>")
            else:
                file.write("<!DOCTYPE html>\n<html>\n" +
                       "\t<table border=\"1\" style=\"margin-left: auto; margin-right: auto;\">\n" +
                       "\t\t<td style=\"padding: 10px;\"><a href=\"" + prevName + "\" target=\"_top\">Previous</a></td>\n" +
                       "\t\t<td style=\"padding: 10px;\"><a href=\"" + nextName + "\" target=\"_top\">Next</a></td>\n" +
                       "\t</table>\n" +
                       "</html>")

def writeKEGGpdf(outputDirectory, organism, superpathway, superpathwayDict, r2gr, r2grall, c2gc, c2gcall):
    """
    """
    index = []
    safeSuperpathway = makesafe(superpathway)
    for pw in superpathwayDict[superpathway]:
        try:
            pathway = KGML_parser.read(kegg_get(organism + pw[0], "kgml"))
        except:
            print("Pathway", pw, "could not be downloaded: ignoring")
            continue
        canvas = KGMLCanvas(pathway)
        for k in pathway.entries:
            t = pathway.entries[k].type
            pathway.entries[k].graphics[0].bgcolor = "#FFFFFF"
            if t == "gene" or t == "ortholog":
                reaction = "rn" + pathway.entries[k].reaction[3:]
                if reaction in r2gr:
                    pathway.entries[k].graphics[0].bgcolor = getColorFor(r2grall[reaction])
                    pathway.entries[k].graphics[0].name = "[R{}] {}".format(r2gr[reaction], pathway.entries[k].graphics[0].name)
            elif t == "compound":
                compound = pathway.entries[k].name[4:]
                if compound in c2gc:
                    pathway.entries[k].graphics[0].bgcolor = getColorFor(c2gcall[compound])
                    pathway.entries[k].graphics[0].name = "[C{}] {}".format(c2gc[compound], pathway.entries[k].graphics[0].name)
        canvas.import_imagemap = True
        pdfName = organism + pw[0] + ".pdf"
        canvas.draw(os.path.join(outputDirectory, organism, safeSuperpathway, pdfName))
        index += [(pdfName, pw[1])]
    return index


def writeIndex(directory, index, title):
    with open(os.path.join(directory, "index.html"), "w") as file:
        file.write("<!DOCTYPE html>\n<html>\n")
        file.write(f"\t<h1>{title}</h1>\n")
        file.write("\t<ul>\n")
        for link, anchor in index:
            file.write("\t\t<li><a href=\"{}\">{}</a>\n".format(link, anchor))
        file.write("\t</ul>\n")
        file.write("</html>\n")



def nafis(d):
    elements = set(dd.keys())
    n = len(elements)
    classes = set(dd.values())
    afiber = n / len(classes)
    
    return afiber / (n - 1)

