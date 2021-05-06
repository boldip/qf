import unittest
import networkx as nx
from numpy import random

import qf.graphs


class TestGraphs(unittest.TestCase):

    def test_add_edges_with_names(self):
        G = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(G, [(0, 1, "a"), (1, 2, "b"), (1, 3, "c"), (1, 4, "d"), (3, 0, "e"), (4, 0, "f")])
        self.assertEqual(5, G.number_of_nodes())
        self.assertEqual(6, G.number_of_edges())
        self.assertTrue(G.has_edge(0, 1))
        self.assertTrue(G.has_edge(1, 2))
        self.assertTrue(G.has_edge(1, 3))
        self.assertTrue(G.has_edge(4, 0))
        print("***", G.get_edge_data(4, 0))
        self.assertEqual("f", G.get_edge_data(4, 0)[0]["label"])

    def test_remove_edges_with_names(self):
        G = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(G, [(0, 1, "a"), (1, 2, "b"), (1, 3, "c"), (1, 4, "d"), (3, 0, "e"), (4, 0, "f")])
        self.assertEqual(5, G.number_of_nodes())
        self.assertEqual(6, G.number_of_edges())
        qf.graphs.removeEdgesWithNames(G, ["a", "e"])
        self.assertEqual(5, G.number_of_nodes())
        self.assertEqual(4, G.number_of_edges())
        self.assertFalse(G.has_edge(0, 1))
        self.assertTrue(G.has_edge(1, 2))
        self.assertTrue(G.has_edge(1, 3))
        self.assertTrue(G.has_edge(4, 0))
        self.assertFalse(G.has_edge(3, 0))
        
    
if __name__ == "__main__": unittest.main()
