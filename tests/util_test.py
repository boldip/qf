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

    def test_read_graph(self):
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
        

if __name__ == "__main__": unittest.main()
