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

    def test_zssDist(self):
        G = nx.MultiDiGraph()
        # Union of bouquets with the following degrees
        deg = [1, 2, 4, 5]
        n = len(deg)
        count = 0
        for x in range(n):
            for i in range(deg[x]):
                qf.graphs.addEdgesWithName(G, [(x, x, "a_{}_{}".format(x, count))])
                count += 1
        M, nodes, indices = qf.qzss.cachedZssDistMatrix(G, 2)
        res = {}
        for cl in range(1, 5):
            clustering, _M, _nodes, _indices = qf.qzss.agclust(G, 2, cl)
            self.assertEqual(set(nodes), set(_nodes))
            for x in nodes:
                for y in nodes:
                    self.assertEqual(M[indices[x], indices[y]], _M[_indices[x], _indices[y]])
            res[cl] = clustering.labels_
        # 1 cluster
        for x in nodes:
            for y in nodes:
                self.assertEqual(res[1][indices[x]], res[1][indices[y]])
        # 2 clusters
        self.assertEqual(res[2][indices[0]], res[2][indices[1]])
        self.assertEqual(res[2][indices[2]], res[2][indices[3]])
        self.assertNotEqual(res[2][indices[0]], res[2][indices[2]])
        # 3 clusters (not clear how merging)
        # 4 clusters        
        self.assertNotEqual(res[4][indices[0]], res[4][indices[1]])
        self.assertNotEqual(res[4][indices[0]], res[4][indices[2]])
        self.assertNotEqual(res[4][indices[0]], res[4][indices[3]])
        # Varying number of clusters: optimal silhouette is for 2
        c, _M, _nodes, _indices = qf.qzss.agclustOptcl(G, 2, 1, 5)
        d = qf.qzss.agclust2dict(c, _M, _nodes, _indices)
        self.assertEqual(set([0, 1, 2, 3]), d.keys())
        self.assertEqual(d[0], d[1])
        self.assertEqual(d[2], d[3])
        self.assertNotEqual(d[0], d[2])

if __name__ == "__main__": unittest.main()
