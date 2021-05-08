import math
import random
import unittest

import networkx as nx
import networkx.linalg.graphmatrix as nxg
import qf.graphs
import qf.morph


class TestGraphs(unittest.TestCase):

    def test_source_target_arcs_new(self):
        G = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(G, [(0, 1, "a"), (1, 2, "b"), (1, 3, "c"), (1, 4, "d"), (3, 0, "e"), (4, 0, "f")])
        self.assertEqual(0, qf.morph.source(G, "a"))
        self.assertEqual(1, qf.morph.source(G, "b"))
        self.assertEqual(3, qf.morph.target(G, "c"))
        self.assertEqual(4, qf.morph.target(G, "d"))
        self.assertEqual(set(["a", "b", "c", "d", "e", "f"]), qf.morph.arcs(G))
        s = qf.morph.arcs(G)
        for i in range(10):
            lab = qf.morph.new_arc_label(G)
            self.assertFalse(lab in s)
            qf.graphs.addEdgesWithName(G, [(random.randint(0, 5), random.randint(0, 5), lab)])
            s |= set([lab])

    def test_morph(self):
        back_forth = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(back_forth, [(0, 1, "a"), (1, 0, "b")])
        single_loop = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(single_loop, [(0, 0, "a")])
        double_loop = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(double_loop, [(0, 0, "a"), (0, 0, "b")])
        double_loop_and_node = nx.MultiDiGraph()
        qf.graphs.addEdgesWithName(double_loop_and_node, [(0, 0, "a"), (0, 0, "b")])
        double_loop_and_node.add_node(1)
        self.assertTrue(qf.morph.is_morphism({0: 0, 1: 1, "a": "a", "b": "b"}, back_forth, back_forth))
        self.assertFalse(qf.morph.is_morphism({0: 0, 1: 1, "a": "b", "b": "a"}, back_forth, back_forth))  # Incompatible
        self.assertFalse(qf.morph.is_morphism({1: 1, "a": "a", "b": "b"}, back_forth, back_forth)) #  Missing node
        self.assertFalse(qf.morph.is_morphism({0: 0, 1: 1, "a": "a"}, back_forth, back_forth)) # Missing arc
        self.assertTrue(qf.morph.is_epimorphism({0: 0, 1: 0, "a": "a", "b": "a"}, back_forth, single_loop))
        self.assertTrue(qf.morph.is_morphism({0: 0, 1: 0, "a": "a", "b": "b"}, back_forth, double_loop))
        self.assertTrue(qf.morph.is_morphism({0: 0, 1: 0, "a": "a", "b": "a"}, back_forth, double_loop))
        # Epi
        self.assertTrue(qf.morph.is_epimorphism({0: 0, 1: 1, "a": "a", "b": "b"}, back_forth, back_forth))
        self.assertFalse(qf.morph.is_epimorphism({0: 0, 1: 1, "a": "b", "b": "a"}, back_forth, back_forth))  # Not a morphism
        self.assertFalse(qf.morph.is_epimorphism({1: 1, "a": "a", "b": "b"}, back_forth, back_forth)) # Not a morphism
        self.assertFalse(qf.morph.is_epimorphism({0: 0, 1: 1, "a": "a"}, back_forth, back_forth)) # Not a morphism
        self.assertTrue(qf.morph.is_epimorphism({0: 0, 1: 0, "a": "a", "b": "a"}, back_forth, single_loop))
        self.assertFalse(qf.morph.is_epimorphism({0: 0, 1: 0, "a": "a", "b": "a"}, back_forth, double_loop)) # Not epi on arcs
        self.assertTrue(qf.morph.is_epimorphism({0: 0, 1: 0, "a": "a", "b": "b"}, back_forth, double_loop)) 
        self.assertFalse(qf.morph.is_epimorphism({0: 0, 1: 0, "a": "a", "b": "b"}, back_forth, double_loop_and_node)) # Not epi on nodes

if __name__ == "__main__": unittest.main()
