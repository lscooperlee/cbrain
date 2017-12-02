import unittest
import numpy as np
import itertools
from numpy.testing import assert_array_equal

from cneuron.networks import RBM
from cneuron.networks import learning_rate
from cneuron.networks import weight_decay
from cneuron.networks import momentum
from cneuron.functions import Logsig
from cneuron.functions import Line
from cneuron.dataset import ProbabilityDataGenerator
from cneuron.dataset import FunctionDataSet

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
        v1 = rbm.sample(v)

        assert_array_equal(len(v1), 3)

    def test_generate(self):
        rbm = RBM((3, 2))
        rbm.T['W'] = np.array([[-1, -2], [-2, 1], [1, 0.5]])
        rbm.T['B'] = np.array([1, -1, 0.5])
        rbm.T['C'] = np.array([1, -2])

        g = rbm.generate(size=100)

    def test_decor(self):

        @learning_rate()
    #    @weight_decay()
    #    @momentum()
        class SimpleBernoulliRBM(RBM):
            pass

        network = SimpleBernoulliRBM((2, 1))
        dg = ProbabilityDataGenerator(pattern = [[0, 0], [1, 1]],
                                      prob = [0.5, 0.5])

        data = dg.get(0, 1000, 1)
        train_iter = network.train(data)

        train5 = itertools.islice(train_iter, 5)
        for s in train5:
            pass

if __name__ == "__main__":
    unittest.main()
