import math
import random
import unittest

import networkx as nx
import networkx.linalg.graphmatrix as nxg
import qf.graphs
import qf.morph
import qf.qzss
import zss


class TestZss(unittest.TestCase):

    def test_zss(self):
        two_bouquet = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(two_bouquet, [(0, 0, "a"), (0, 0, "b")])
        for k in range(3):
            it = qf.qzss.allPaths(two_bouquet, 0, k) #2^(k+1)-1 paths are returned
            self.assertEqual((2**(k + 1) - 1), sum(1 for _ in it))

    def test_inTree(self):
        two_bouquet = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(two_bouquet, [(0, 0, "a"), (0, 0, "b")])
        for k in range(3):
            T = qf.qzss.inTree(two_bouquet, 0, k) #an intree with 2^k leaves and 2^(k+1)-1 nodes
            self.assertEqual((2**(k + 1) - 1), T.number_of_nodes())
            count_leaves = 0
            for x in T.nodes():
                if len(T.in_edges(x)) == 0:
                    count_leaves += 1
            self.assertEqual(2**k, count_leaves)

    def is_binary_intree_of_height(T, k):
        if k == 0:
            return isinstance(T, zss.Node) and len(T.getChildren()) == 0
        if not isinstance(T, zss.Node):
            return False
        print(type(T.getChildren()))
        if len(T.getChildren()) != 2:
            return False
        

    def test_zssAllPaths(self):
        two_bouquet = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(two_bouquet, [(0, 0, "a"), (0, 0, "b")])
        for k in range(3):
            T = qf.qzss.zssAllPaths(two_bouquet, 0, k) #an intree with 2^k leaves and 2^(k+1)-1 nodes
            is_binary_intree_of_height(T, k)

if __name__ == "__main__": unittest.main()
