import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from cneuron.networks import RBM


class TestRBM(unittest.TestCase):

    def test_RBM_constructor(self):
        rbm = RBM(3, 2)
        print(rbm.B, rbm.W, rbm.C)

    def test_RBM_energy(self):
        v = np.array([1, 1, 0])
        h = np.array([1, 0])

        w = np.array([[0.1, 0.1], [0.2, 0], [-0.1, 0]])
        b = np.array([1, 1, 0])
        c = np.array([0, 1])

        rbm = RBM(3, 2, W=w, B=b, C=c)
        ret = rbm.E(v, h)
        print(ret)

    def test_RBM_P(self):
        v = np.array([1, 1, 0])
        h = np.array([1, 0])

        w = np.array([[0.1, 0.1], [0.2, 0], [-0.1, 0]])
        b = np.array([1, 1, 0])
        c = np.array([0, 1])

        rbm = RBM(3, 2, W=w, B=b, C=c)
        ret = rbm.Z()
        #print(ret)

        #ret = rbm.P(v, h)
        #print(ret)

        import itertools
        all_v = itertools.product([0, 1], repeat=3)
        all_h = itertools.product([0, 1], repeat=2)

        for v, h in itertools.product(all_v, all_h):
            print(v, h, rbm.P(v, h))

if __name__ == "__main__":
    unittest.main()
