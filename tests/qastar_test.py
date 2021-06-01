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

if __name__ == "__main__": unittest.main()
