import unittest

import networkx as nx
import qf.cc
from numpy import random


class TestCc(unittest.TestCase):

    def test_cardon_crochemore(self):
        G = nx.MultiDiGraph()
        G.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (3, 0), (4, 0)])
        cc = qf.cc.cardon_crochemore(G)
        self.assertEqual(3, len(set(cc.values())))
        self.assertEqual(set([0, 1, 2, 3, 4]), cc.keys())
        self.assertEqual(cc[3], cc[4])
        cc = qf.cc.cardon_crochemore(G, max_step = 1) #indegree only
        self.assertEqual(2, len(set(cc.values())))
        self.assertEqual(set([0, 1, 2, 3, 4]), cc.keys())
        self.assertEqual(cc[3], cc[4])
        self.assertNotEqual(cc[0], cc[4])
        
    def test_cardon_crochemore_inregular(self):
        G = nx.MultiDiGraph()
        k = 5
        n = 100
        for t in range(n):
            sources = random.choice(n, k)
            for s in sources:
                G.add_edge(s, t)
        cc = qf.cc.cardon_crochemore(G)
        self.assertEqual(1, len(set(cc.values())))

if __name__ == "__main__": unittest.main()
