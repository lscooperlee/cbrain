import unittest
import numpy as np
import itertools
from numpy.testing import assert_array_equal

from cbrain.network import RBM
from cbrain.network import learning_rate
from cbrain.network import weight_decay
from cbrain.network import momentum
from cbrain.function import Logsig
from cbrain.function import Line
from cbrain.dataset import ProbabilityDataGenerator
from cbrain.dataset import FunctionDataSet

class ProbFunctionDataSet(FunctionDataSet, Line, ProbabilityDataGenerator):
    pass

class TestRBM(unittest.TestCase):

    def test_RBM(self):
        rbm = RBM((3, 2))
        self.assertEqual(len(rbm.T['B']), 3)
        self.assertEqual(len(rbm.T['C']), 2)
        self.assertEqual(len(rbm.T['W']), 3)

    def test_alpha(self):
        rbm = RBM((3, 2))
        rbm.T['W'] = np.array([[1, 2], [2, 1], [1, 0.5]])
        rbm.T['B'] = np.array([1, 1, 0.5])
        rbm.T['C'] = np.array([1, 2])

        v = np.array([1, 1, 0])
        assert_array_equal(rbm._alpha(v), [4, 5])

    def test_beta(self):
        rbm = RBM((3, 2))
        rbm.T['W'] = np.array([[1, 2], [2, 1], [1, 0.5]])
        rbm.T['B'] = np.array([1, 1, 0.5])
        rbm.T['C'] = np.array([1, 2])

        h = np.array([0, 1])
        assert_array_equal(rbm._beta(h), [3, 2, 1])

    def test_train_iter(self):
        rbm = RBM((3, 2))
        rbm.T['W'] = np.array([[1, 2], [2, 1], [1, 0.5]])
        rbm.T['B'] = np.array([1, 1, 0.5])
        rbm.T['C'] = np.array([1, 2])

        v = np.array([0, 1, 1])
        rbm.CD(v)

    def test_sample(self):
        rbm = RBM((3, 2))
        rbm.T['W'] = np.array([[-1, -2], [-2, 1], [1, 0.5]])
        rbm.T['B'] = np.array([1, -1, 0.5])
        rbm.T['C'] = np.array([1, -2])

        v = np.array([0, 1, 0])
        v1, _ = rbm.sample(v)

        assert_array_equal(len(v1), 3)

    def test_generate(self):
        rbm = RBM((3, 2))
        rbm.T['W'] = np.array([[-1, -2], [-2, 1], [1, 0.5]])
        rbm.T['B'] = np.array([1, -1, 0.5])
        rbm.T['C'] = np.array([1, -2])

        g = rbm.generate()

    def test_decor(self):

        @learning_rate()
        @weight_decay()
        @momentum()
        class SimpleBernoulliRBM(RBM):
            pass

        network = SimpleBernoulliRBM((3, 2))
        dg = ProbabilityDataGenerator(pattern = [[0, 0, 1], [1, 1, 0]],
                                      prob = [0.5, 0.5])

        data = dg.get(0, 10, 1)

        train_iter = network.train(data)

        train5 = itertools.islice(train_iter, 5)
        for s in train5:
            pass

if __name__ == "__main__":
    unittest.main()
