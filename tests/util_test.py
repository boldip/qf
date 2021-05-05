import tempfile
import unittest

import qf.util


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
        res = qf.util.readGraph(tempFile.name, skipHeader=True, separator=",", dense=False)
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
        res = qf.util.readGraph(tempFile.name, skipHeader=False, separator="\t", dense=False)
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
        res = qf.util.readGraph(tempFile.name, skipHeader=True, separator=",", dense=True)
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
        res = qf.util.readLabel(tempFile.name, skipHeader=True, separator=",")
        self.assertEqual({"A": "0", "B": "0", "C": "1", "D": "0"}, res)
        tempFile = tempfile.NamedTemporaryFile()
        with open(tempFile.name, "w") as txt:
            txt.write(
                """A\t0
                B\t0
                C\t1
                D\t0
                """)
        res = qf.util.readLabel(tempFile.name, skipHeader=False, separator="\t")
        self.assertEqual({"A": "0", "B": "0", "C": "1", "D": "0"}, res)

    def test_read_label(self):
        tempFile = tempfile.NamedTemporaryFile()
        with open(tempFile.name, "w") as txt:
            txt.write(
                """# Ignore this
                A,0,0.1
                B,2,1.2
                C,0,0
                D,11112.1,1
                """)
        res = qf.util.readCoordinates(tempFile.name, skipHeader=True, separator=",")
        self.assertEqual({"A": (0,0.1), "B": (2,1.2), "C": (0,0), "D": (11112.1,1)}, res)
        tempFile = tempfile.NamedTemporaryFile()
        with open(tempFile.name, "w") as txt:
            txt.write(
                """A\t0\t0.1
                B\t2\t1.2
                C\t0\t0
                D\t11112.1\t1
                """)
        res = qf.util.readCoordinates(tempFile.name, skipHeader=False, separator="\t")
        self.assertEqual({"A": (0,0.1), "B": (2,1.2), "C": (0,0), "D": (11112.1,1)}, res)

    def test_colors(self):
        self.assertEquals(10, qf.util.colors(10))
        
if __name__ == "__main__": unittest.main()
