import unittest
import networkx as nx

import qf.cc


class TestUtil(unittest.TestCase):

    def test_cardon_crochemore(self):
        G = nx.MultiDiGraph()
        G.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (3, 0), (4, 0)])
        cc = qf.cc.cardon_crochemore(G)
        self.assertEquals(3, len(set(cc.values())))
        self.assertEquals(set([0, 1, 2, 3, 4]), cc.keys())
        self.assertEquals(cc[3], cc[4])
        
if __name__ == "__main__": unittest.main()
