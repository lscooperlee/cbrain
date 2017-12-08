import unittest
import numpy as np
import itertools
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_less

from cbrain.network import FFN
from cbrain.network import learning_rate
from cbrain.network import weight_decay
from cbrain.network import momentum
from cbrain.function import Logsig
from cbrain.function import Line
from cbrain.dataset import FunctionDataSet
from cbrain.dataset import ProbabilityDataGenerator
from cbrain.dataset import SequenceDataGenerator
from cbrain.dataset import IntegerDataGenerator
from cbrain.dataset import RandomDataGenerator


class TestFFN(unittest.TestCase):

    def test_FFN(self):
        net = FFN((3, 2, 1))

    def test_train_iter(self):
        net = FFN((1, 2, 1))

        net.T["W01"] = np.array([[3, 1]])
        net.T["W12"] = np.array([[2], [1]])
        net.T["B1"] = np.array([1, 1])
        net.T["B2"] = np.array([1])

        training_data = np.array([1])
        testing_data = np.array([12])

        dT = net.train_iter((training_data, testing_data))
        for v in dT.values():
            assert_array_less(v, np.zeros(v.shape))

    def test_forward(self):
        net = FFN((1, 2, 1))
        net.T["W01"] = np.array([[3, 1]])
        net.T["W12"] = np.array([[2], [1]])
        net.T["B1"] = np.array([1, 1])
        net.T["B2"] = np.array([1])
        net.F = Line()

        dg = IntegerDataGenerator()
        idata = dg.get(1, 3, 1)
        odata = np.array([net.forward(v) for v in idata])

        assert_array_equal(odata, np.array([[11], [18]]))

    def test_train(self):
        net = FFN((1, 2, 1))
        net.T["W01"] = np.array([[3, 1]])
        net.T["W12"] = np.array([[2], [1]])
        net.T["B1"] = np.array([1, 1])
        net.T["B2"] = np.array([1])

        class LineSequenceDataSet(FunctionDataSet, Line, SequenceDataGenerator):
            pass

        seq = (0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
        dataset = LineSequenceDataSet(sequence=seq, T=2)
        training, testing = dataset.train_data(0, len(seq))

        train_iter = net.train(zip(training, testing))

        train5 = itertools.islice(train_iter, 5)
        for s in train5:
            pass


if __name__ == "__main__":
    unittest.main()
