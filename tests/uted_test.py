import tempfile
import unittest

import uted.uted


class TestUtil(unittest.TestCase):

    def test_uted(self):
        nd = ["x" for i in range(6)]
        t1 = [[1,3], [2], [], [4,5], [], []]   # x(x(x),x(x,x))
        t2 = [[1,4], [2,3], [], [], [5], []]   # x(x(x,x),x(x))
        self.assertEquals(0.0, uted.uted.uted_astar(nd, t1, nd, t2)[0])

if __name__ == "__main__": unittest.main()
