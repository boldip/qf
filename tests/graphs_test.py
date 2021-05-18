import math
import random
import unittest

import networkx as nx
import networkx.linalg.graphmatrix as nxg
import qf.graphs
import graphviz 


class TestGraphs(unittest.TestCase):

    def test_add_edges_with_names(self):
        G = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(G, [(0, 1, "a"), (1, 2, "b"), (1, 3, "c"), (1, 4, "d"), (3, 0, "e"), (4, 0, "f")])
        self.assertEqual(5, G.number_of_nodes())
        self.assertEqual(6, G.number_of_edges())
        self.assertTrue(G.has_edge(0, 1))
        self.assertTrue(G.has_edge(1, 2))
        self.assertTrue(G.has_edge(1, 3))
        self.assertTrue(G.has_edge(4, 0))
        self.assertEqual("f", G.get_edge_data(4, 0)[0]["label"])

    def test_remove_edges_with_names(self):
        G = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(G, [(0, 1, "a"), (1, 2, "b"), (1, 3, "c"), (1, 4, "d"), (3, 0, "e"), (4, 0, "f")])
        self.assertEqual(5, G.number_of_nodes())
        self.assertEqual(6, G.number_of_edges())
        qf.graphs.remove_edges_with_name(G, ["a", "e"])
        self.assertEqual(5, G.number_of_nodes())
        self.assertEqual(4, G.number_of_edges())
        self.assertFalse(G.has_edge(0, 1))
        self.assertTrue(G.has_edge(1, 2))
        self.assertTrue(G.has_edge(1, 3))
        self.assertTrue(G.has_edge(4, 0))
        self.assertFalse(G.has_edge(3, 0))

    def test_lift(self):
        G = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(G, [(0, 1, "a"), (1, 2, "b"), (2, 0, "d1"), (2, 0, "d2")])
        H = qf.graphs.lift(G, {0: [0], 1: [0], 2: [6, 7]})
        self.assertEqual(4, H.number_of_nodes())
        self.assertEqual(5, H.number_of_edges())
        self.assertEqual(set([(0,0), (1,0), (2,6), (2,7)]), set(H.nodes()))
        for x in H.nodes:
            self.assertEqual(G.in_degree(x[0]), H.in_degree(x))
            for s,t in H.in_edges(x):
                self.assertTrue(G.has_edge(s[0], x[0]))

    def test_minimum_base(self):
        G = nx.MultiDiGraph()
        n1 = 50
        n2 = 20
        k1 = 4
        k2 = 2
        count = 0
        # n1 nodes have k1+1 incoming arcs (k1 from the same group, 1 from the other)
        G.add_nodes_from(["a" +  str(x) for x in range(n1)])
        for x in range(n1):
            s = random.sample(range(n1), k1)
            for y in s:
                qf.graphs.add_edges_with_name(G, [("a" + str(y), "a" + str(x), "arcaa_" + str(y) + "_" + str(x) + "_" + str(count))])
                count += 1
        # n2 nodes have k2+2 incoming arcs (k2 from the same group, 2 from the other)
        G.add_nodes_from(["b" +  str(x) for x in range(n2)])
        for x in range(n2):
            s = random.sample(range(n2), k2)
            for y in s:
                qf.graphs.add_edges_with_name(G, [("b" + str(y), "b" + str(x), "arcbb_" + str(y) + "_" + str(x) + "_" + str(count))])
                count += 1
        # 
        for x in range(n1):
            ss = random.sample(range(n2), 1)
            qf.graphs.add_edges_with_name(G, [("b" + str(ss[0]), "a" + str(x), "arcba_" + str(ss[0]) + "_" + str(x) + "_" + str(count))])
            count += 1
        # 
        for x in range(n2):
            ss = random.sample(range(n1), 2)
            qf.graphs.add_edges_with_name(G, [("a" + str(ss[0]), "b" + str(x), "arcab_" + str(ss[0]) + "_" + str(x) + "_" + str(count))])
            count += 1
            qf.graphs.add_edges_with_name(G, [("a" + str(ss[1]), "b" + str(x), "arcab_" + str(ss[1]) + "_" + str(x) + "_" + str(count))])
            count += 1
        # Now compute the minimum base
        B = qf.graphs.minimum_base(G, qf.cc.cardon_crochemore(G))
        self.assertEqual(2, B.number_of_nodes())
        self.assertEqual(k1 + k2 + 2 + 1, B.number_of_edges())
        for x in B.nodes():
            if len(B.in_edges(x)) == k1 + 1:
                x1 = x
            else:
                x2 = x
        self.assertEqual(k1, B.number_of_edges(x1, x1))
        self.assertEqual(k2, B.number_of_edges(x2, x2))
        self.assertEqual(2, B.number_of_edges(x1, x2))
        self.assertEqual(1, B.number_of_edges(x2, x1))

    def test_scramble(self):
        G = nx.MultiDiGraph()
        n = 50
        count = 0
        for x in range(n):
            k = random.randrange(0, n//2)
            out = random.sample(range(n), k)
            for y in out:
                qf.graphs.add_edges_with_name(G, [("a" + str(x), "a" + str(y), "c" + str(count))])
                count += 1
        x1 = random.randrange(0, n//5)
        x2 = random.randrange(0, n//5)
        x3 = random.randrange(0, n//5)
        deleted = 0
        added = 0
        H = qf.graphs.scramble(G, x1, x2, x3)
        self.assertEqual(set(G.nodes()), set(H.nodes()))
        for x in G.nodes():
            for y in G.nodes():
                tg = G.number_of_edges(x, y)
                th = H.number_of_edges(x, y)
                if tg > th:
                    deleted += tg - th
                else:
                    added += th - tg
        self.assertEqual(x1 + x2 + x3, deleted + added)
        self.assertTrue(added >= x1)
        self.assertTrue(deleted >= x2)

    def test_to_simple(self):
        G = nx.MultiDiGraph()
        n = 50
        count = 0
        for x in range(n):
            k = random.randrange(0, n//2)
            out = random.sample(range(n), k)
            for y in out:
                qf.graphs.add_edges_with_name(G, [("a" + str(x), "a" + str(y), "c" + str(count))])
                count += 1
        H = qf.graphs.to_simple(G)
        self.assertEqual(set(H.nodes()), set(G.nodes()))
        for x in G.nodes():
            for y in G.nodes():
                tg = G.number_of_edges(x, y)
                th = H.number_of_edges(x, y)
                self.assertTrue((tg > 0) == (th == 1))

    def test_difference(self):
        G = nx.MultiDiGraph()
        H = nx.MultiDiGraph()
        n = 50
        count = 0
        for x in range(n):
            k = random.randrange(0, n//2)
            out = random.sample(range(n), k)
            for y in out:
                qf.graphs.add_edges_with_name(G, [("a" + str(x), "a" + str(y), "c" + str(count))])
                count += 1
        for x in range(n):
            k = random.randrange(0, n//2)
            out = random.sample(range(n), k)
            for y in out:
                qf.graphs.add_edges_with_name(H, [("a" + str(x), "a" + str(y), "c" + str(count))])
                count += 1
        D = qf.graphs.difference(G, H)
        for x in range(n):
            for y in range(n):
                tg = G.number_of_edges(x, y)
                th = H.number_of_edges(x, y)
                td = D.number_of_edges(x, y)
                self.assertEqual(abs(tg - th), td)

    def test_save(self):
        G = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(G, [(0, 1, "a"), (1, 2, "b"), (1, 3, "c"), (1, 4, "d"), (3, 0, "e"), (4, 0, "f")])
        dotf, pngf = qf.graphs.save(G)
        Gread = graphviz.Digraph(filename=dotf)
        print(Gread)


if __name__ == "__main__": unittest.main()
