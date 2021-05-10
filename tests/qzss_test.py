import math
from random import randrange
from numpy import random
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

    def is_binary_intree(self, T, k):
        if k == 0:
            return isinstance(T, zss.Node) and len(T.get_children(T)) == 0
        if not isinstance(T, zss.Node):
            return False
        if len(T.get_children(T)) != 2:
            return False
        return self.is_binary_intree(T.get_children(T)[0], k - 1) and self.is_binary_intree(T.get_children(T)[1], k - 1)
        
    def test_zssAllPaths(self):
        two_bouquet = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(two_bouquet, [(0, 0, "a"), (0, 0, "b")])
        for k in range(3):
            T = qf.qzss.zssAllPaths(two_bouquet, 0, k) #an intree with 2^k leaves and 2^(k+1)-1 nodes
            self.assertTrue(self.is_binary_intree(T, k))

    def test_zssDist(self):
        back_forth = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(back_forth, [(0, 1, "a"), (1, 0, "b"), (0, 0, "c")])
        self.assertEqual(4.0, qf.qzss.zssTreeDist(back_forth, 0, 1, 3))
        self.assertEqual(0.0, qf.qzss.zssTreeDist(back_forth, 0, 0, 3))

    def test_zssDistIn(self):
        G = nx.MultiDiGraph()
        n = 20
        k = 3
        for x in range(n):
            G.add_node(x)
        count = 0
        for x in range(n):
            sources = random.choice(n, k)
            for source in sources:
                qf.graphs.addEdgesWithName(G, [(source, x, "a_{}".format(count))])
                count += 1
        for t in range(10):
            x = randrange(0, n)
            y = randrange(0, n)
            d = randrange(0, 3)
            self.assertEqual(0.0, qf.qzss.zssTreeDist(G, x, y, d))

    def test_zssDist(self):
        back_forth = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(back_forth, [(0, 1, "a"), (1, 0, "b"), (0, 0, "c")])
        M, nodes, indices = qf.qzss.cachedZssDistMatrix(back_forth, 3)
        self.assertEqual(0, M[indices[0], indices[0]])
        self.assertEqual(0, M[indices[1], indices[1]])
        self.assertEqual(4, M[indices[0], indices[1]])
        self.assertEqual(4, M[indices[1], indices[0]])


if __name__ == "__main__": unittest.main()
