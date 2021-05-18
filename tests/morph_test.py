import math
import random
import unittest

import networkx as nx
import networkx.linalg.graphmatrix as nxg
import qf.graphs
import qf.morph


class TestMorph(unittest.TestCase):

    def test_source_target_arcs_new(self):
        G = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(G, [(0, 1, "a"), (1, 2, "b"), (1, 3, "c"), (1, 4, "d"), (3, 0, "e"), (4, 0, "f")])
        self.assertEqual(0, qf.morph.source(G, "a"))
        self.assertEqual(1, qf.morph.source(G, "b"))
        self.assertEqual(3, qf.morph.target(G, "c"))
        self.assertEqual(4, qf.morph.target(G, "d"))
        self.assertEqual(set(["a", "b", "c", "d", "e", "f"]), qf.morph.arcs(G))
        s = qf.morph.arcs(G)
        for i in range(10):
            lab = qf.morph.new_arc_label(G)
            self.assertFalse(lab in s)
            qf.graphs.add_edges_with_name(G, [(random.randint(0, 5), random.randint(0, 5), lab)])
            s |= set([lab])

    def test_morph(self):
        back_forth = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(back_forth, [(0, 1, "a"), (1, 0, "b")])
        single_loop = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(single_loop, [(0, 0, "a")])
        double_loop = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(double_loop, [(0, 0, "a"), (0, 0, "b")])
        double_loop_and_node = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(double_loop_and_node, [(0, 0, "a"), (0, 0, "b")])
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
        # Fibration
        self.assertFalse(qf.morph.is_fibration({0: 0, 1: 1, "a": "b", "b": "a"}, back_forth, back_forth))  # Incompatible

    def test_excess_deficiency(self):
        back_forth = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(back_forth, [(0, 1, "a"), (1, 0, "b")])
        single_loop = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(single_loop, [(0, 0, "a")])
        f = {0: 0, 1: 0, "a": "a", "b": "a"}   # a fibration from back_forth to single_loop
        x, d = qf.morph.excess_deficiency(f, back_forth, single_loop)
        self.assertEqual(0, x)
        self.assertEqual(0, d)
        double_loop = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(double_loop, [(0, 0, "a"), (0, 0, "b")])
        f = {0: 0, 1: 0, "a": "a", "b": "b"}   # two missing liftings from back_forth to double_loop
        x, d = qf.morph.excess_deficiency(f, back_forth, double_loop)
        self.assertEqual(0, x)
        self.assertEqual(2, d)
        f = {0: 0, "a": "a", "b": "a"}  # one excess lifting from double_loop to single_loop
        x, d = qf.morph.excess_deficiency(f, double_loop, single_loop)
        self.assertEqual(1, x)
        self.assertEqual(0, d)


    def test_repair(self):
        back_forth = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(back_forth, [(0, 1, "a"), (1, 0, "b")])
        single_loop = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(single_loop, [(0, 0, "a")])
        f = {0: 0, 1: 0, "a": "a", "b": "a"}   # a fibration from back_forth to single_loop
        Gp, fp = qf.morph.repair(f, back_forth, single_loop)
        self.assertEqual(f, fp)
        self.assertTrue(qf.morph.is_compatible(back_forth, Gp, f, fp))
        self.assertEqual(qf.morph.arcs(back_forth), qf.morph.arcs(Gp))
        double_loop = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(double_loop, [(0, 0, "a"), (0, 0, "b")])
        f = {0: 0, 1: 0, "a": "a", "b": "b"}   # two missing liftings from back_forth to double_loop
        Gp, fp = qf.morph.repair(f, back_forth, double_loop)
        self.assertTrue(qf.morph.is_compatible(back_forth, Gp, f, fp))
        self.assertEqual(2, len(qf.morph.arcs(Gp) - qf.morph.arcs(back_forth)))
        self.assertEqual(0, len(qf.morph.arcs(back_forth) - qf.morph.arcs(Gp)))
        self.assertTrue(qf.morph.is_fibration(fp, Gp, double_loop))
        f = {0: 0, "a": "a", "b": "a"}  # one excess lifting from double_loop to single_loop
        Gp, fp = qf.morph.repair(f, double_loop, single_loop)
        self.assertTrue(qf.morph.is_compatible(double_loop, Gp, f, fp))
        self.assertEqual(0, len(qf.morph.arcs(Gp) - qf.morph.arcs(double_loop)))
        self.assertEqual(1, len(qf.morph.arcs(double_loop) - qf.morph.arcs(Gp)))
        self.assertTrue(qf.morph.is_fibration(fp, Gp, single_loop))

    def test_qf_build(self):
        two_one = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(two_one, [(0, 1, "a1"), (0, 1, "a2"), (1, 0, "b")])
        B, f = qf.morph.qf_build(two_one, {0: 0, 1: 1})  # Returns an isomorphic graph
        self.assertTrue(qf.morph.is_isomorphism(f, two_one, B))
        two_one = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(two_one, [(0, 1, "a1"), (0, 1, "a2"), (1, 0, "b")])
        B, f = qf.morph.qf_build(two_one, {0: 0, 1: 0})  # A loop 
        self.assertTrue(1, B.number_of_nodes())
        self.assertTrue(2, B.number_of_edges())
        qf.morph.is_epimorphism(f, two_one, B)

if __name__ == "__main__": unittest.main()
