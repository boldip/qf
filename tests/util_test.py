import networkx as nx
import tempfile
import unittest

import qf.util
import qf.graphs

import qf.uted.uted


class TestUtil(unittest.TestCase):

    def test_indexify(self):
        res = qf.util.indexify({0: "a", 1: "a", 2: "b", 3: ["a", "b"], 4: ["a", "b"]})
        self.assertEqual(res[0], res[1])
        self.assertEqual(res[3], res[4])
        self.assertNotEqual(res[0], res[2])
        self.assertNotEqual(res[0], res[3])
        self.assertNotEqual(res[2], res[3])
        self.assertEqual(set([0, 1, 2, 3, 4]), res.keys())

    def test_jaccard(self):
        a = [0, 0, 1, 2, 3, 3, 3]
        b = [0, 3, 3, 5, 5, 5]
        # 0: 2 vs. 1
        # 1: 1 vs. 0
        # 2: 1 vs. 0
        # 3: 3 vs. 2
        # 5: 0 vs. 3
        expected = (1 + 0 + 0 + 2 + 0) / (2 + 1 + 1 + 3 + 3)
        self.assertEqual(expected, qf.util.jaccard_multiset(a, b))
        a = []
        b = []
        expected = 1   # Jaccard similarity if the sets are both empty
        self.assertEqual(expected, qf.util.jaccard_multiset(a, b))

    def test_read_graph_sparse(self):
        tempFile = tempfile.NamedTemporaryFile()
        with open(tempFile.name, "w") as txt:
            txt.write(
                """# Ignore this please
                A,B
                A,C
                A,B
                C,B
                """)
        res = qf.util.read_graph(tempFile.name, skipHeader=True, separator=",", dense=False)
        self.assertEqual(set(["A", "B", "C"]), set(res.nodes()))
        self.assertEqual(3, res.number_of_nodes())
        self.assertEqual(4, res.number_of_edges())
        self.assertTrue(res.has_edge("A", "B"))
        self.assertTrue(res.has_edge("A", "C"))
        self.assertTrue(res.has_edge("C", "B"))
        self.assertEqual(2, len(res.get_edge_data("A", "B")))
        with open(tempFile.name, "w") as txt:
            txt.write(
                """A\tB
                A\tC
                A\tB
                C\tB
                """)
        res = qf.util.read_graph(tempFile.name, skipHeader=False, separator="\t", dense=False)
        self.assertEqual(set(["A", "B", "C"]), set(res.nodes()))
        self.assertEqual(3, res.number_of_nodes())
        self.assertEqual(4, res.number_of_edges())
        self.assertTrue(res.has_edge("A", "B"))
        self.assertTrue(res.has_edge("A", "C"))
        self.assertTrue(res.has_edge("C", "B"))
        self.assertEqual(2, len(res.get_edge_data("A", "B")))

    def test_read_graph_dense(self):
        tempFile = tempfile.NamedTemporaryFile()
        with open(tempFile.name, "w") as txt:
            txt.write(
                """Targets,A,B,C
                A,0,2,1
                C,0,1,0
                """)
        res = qf.util.read_graph(tempFile.name, skipHeader=True, separator=",", dense=True)
        self.assertEqual(set(["A", "B", "C"]), set(res.nodes()))
        self.assertEqual(3, res.number_of_nodes())
        self.assertEqual(4, res.number_of_edges())
        self.assertTrue(res.has_edge("A", "B"))
        self.assertTrue(res.has_edge("A", "C"))
        self.assertTrue(res.has_edge("C", "B"))
        self.assertEqual(2, len(res.get_edge_data("A", "B")))

    def test_read_label(self):
        tempFile = tempfile.NamedTemporaryFile()
        with open(tempFile.name, "w") as txt:
            txt.write(
                """# Ignore this
                A,0
                B,0
                C,1
                D,0
                """)
        res = qf.util.read_label(tempFile.name, skipHeader=True, separator=",")
        self.assertEqual({"A": "0", "B": "0", "C": "1", "D": "0"}, res)
        tempFile = tempfile.NamedTemporaryFile()
        with open(tempFile.name, "w") as txt:
            txt.write(
                """A\t0
                B\t0
                C\t1
                D\t0
                """)
        res = qf.util.read_label(tempFile.name, skipHeader=False, separator="\t")
        self.assertEqual({"A": "0", "B": "0", "C": "1", "D": "0"}, res)

    def test_read_coordinates(self):
        tempFile = tempfile.NamedTemporaryFile()
        with open(tempFile.name, "w") as txt:
            txt.write(
                """# Ignore this
                A,0,0.1
                B,2,1.2
                C,0,0
                D,11112.1,1
                """)
        res = qf.util.read_coordinates(tempFile.name, skipHeader=True, separator=",")
        self.assertEqual({"A": (0,0.1), "B": (2,1.2), "C": (0,0), "D": (11112.1,1)}, res)
        with open(tempFile.name, "w") as txt:
            txt.write(
                """A\t0\t0.1
                B\t2\t1.2
                C\t0\t0
                D\t11112.1\t1
                """)
        res = qf.util.read_coordinates(tempFile.name, skipHeader=False, separator="\t")
        self.assertEqual({"A": (0,0.1), "B": (2,1.2), "C": (0,0), "D": (11112.1,1)}, res)

    def test_read_graph_coordinates(self):
        tempFile = tempfile.NamedTemporaryFile()
        with open(tempFile.name, "w") as txt:
            txt.write(
                """# Ignore this please
                A,B
                A,C
                A,B
                C,B
                """)
        coordinates = {"A": (0,0.1), "B": (2,1.2), "C": (0,0), "D": (11112.1,1)}
        scale = 5
        res = qf.util.read_graph(tempFile.name, skipHeader=True, separator=",", dense=False, coordinates=coordinates, scale=scale)
        for node,d in res.nodes(data=True):
            self.assertEqual("{},{}!".format(coordinates[node][0] * scale, coordinates[node][1] * scale), d["pos"])

    def test_nmi(self):
        a={0: 0, 1: 0, 2: 1, 3: 1}
        ap={0: 4, 1: 4, 2: 3, 3: 3}
        self.assertEqual(1.0, qf.util.nmi(a, a))
        self.assertEqual(1.0, qf.util.nmi(a, ap))
        a={0: 0, 1: 0, 2: 0, 3: 0}
        b={0: 0, 1: 1, 2: 2, 3: 3}
        self.assertEqual(0.0, qf.util.nmi(a, b)) # Example from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
    
    def test_colors(self):
        self.assertEqual(10, len(qf.util.colors(10)))

    def test_dfs_tree(self):
        G = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(G, [("a", "b", "ba"), ("b", "a", "ab"), ("c", "a", "ca"), ("d", "c", "dc"), ("e", "c", "ec"), ("a", "d", "ad"), ("a", "e", "ae")])
        n, a=qf.util.dfs_tree(G, "a", 2, 0)
        self.assertEqual(["a", "b", "a", "c", "d", "e"], n)
        self.assertEqual([[1,3], [2], [], [4,5], [], []], a)

    def test_dfs_astar(self):
        G = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(G, [("a", "b", "ba"), ("b", "a", "ab"), ("c", "a", "ca"), ("d", "c", "dc"), ("e", "c", "ec"), ("a", "d", "ad"), ("a", "e", "ae")])
        n, a=qf.util.dfs_tree(G, "a", 2)
        H = nx.MultiDiGraph()
        qf.graphs.add_edges_with_name(H, [("c", "a", "ca"), ("d", "c", "dc"), ("e", "c", "ec"), ("a", "d", "ad"), ("a", "e", "ae"), ("a", "b", "ba"), ("b", "a", "ab")])
        np, ap=qf.util.dfs_tree(H, "a", 2)
        self.assertEqual(["a", "c", "d", "e", "b", "a"], np)
        self.assertEqual([[1, 4], [2, 3], [], [], [5], []], ap)
        self.assertEqual(0.0, qf.uted.uted.uted_astar(n, a, np, ap)[0])


if __name__ == "__main__": unittest.main()
