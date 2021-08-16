import math
from random import randrange
from numpy import random
import unittest

import networkx as nx
import networkx.linalg.graphmatrix as nxg
import qf.graphs
import qf.morph
import qf.qastar


class TestQastar(unittest.TestCase):

    def test_qastarDist(self):
        back_forth = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(back_forth, [(0, 1, "a"), (1, 0, "b"), (0, 0, "c")])
        M, nodes, indices = qf.qastar.qastar_dist_matrix(back_forth, 1)
        self.assertEqual(1.0, M[indices[0], indices[1]])

    def test_qastarDist_zero(self):
        back_forth = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(back_forth, [(0, 1, "a"), (1, 0, "b"), (0, 0, "c")])
        M, nodes, indices = qf.qastar.qastar_dist_matrix(back_forth, 1, zero = {0: "x", 1: "x"})
        self.assertEqual(0, M[indices[0], indices[1]])
    
    def test_qastarDist_coloring(self):
        back_forth = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(back_forth, [(0, 1, "a"), (1, 0, "b"), (0, 0, "c")])
        nodes_zero, a_zero = qf.qastar.qastar_all_paths(back_forth, 0, 1, {0: "red", 1: "blue"})
        self.assertEqual(['red', 'blue', 'red'], nodes_zero)
        self.assertEqual([[1, 2], [], []], a_zero)
        nodes_one, a_one = qf.qastar.qastar_all_paths(back_forth, 1, 1, {0: "red", 1: "blue"})
        self.assertEqual(['blue', 'red'], nodes_one)
        self.assertEqual([[1], []], a_one)
        M, nodes, indices = qf.qastar.qastar_dist_matrix(back_forth, 1, None, {0: "red", 1: "blue"})
        self.assertEqual(2.0, M[indices[0], indices[1]]) # rename and insert
        self.assertEqual(2.0, M[indices[1], indices[0]])
        self.assertEqual(0, M[indices[0], indices[0]])
        self.assertEqual(0, M[indices[1], indices[1]])
    
    def test_agclust(self):
        back_forth = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(back_forth, [(0, 1, "a"), (1, 0, "b"), (0, 0, "c")])
        clustering, M, nodes, indices = qf.qastar.agclust(back_forth, 1, 2)
        agclust2dict = qf.qastar.agclust2dict(clustering, M, nodes, indices)
        self.assertTrue(agclust2dict == {0: 1, 1: 0} or agclust2dict == {0: 0, 1: 1})
    

if __name__ == "__main__": unittest.main()
