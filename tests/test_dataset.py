import unittest
import numpy as np
import itertools
from collections import Counter

from cbrain.dataset import FunctionDataSet
from cbrain.dataset import MnistDataSet
from cbrain.dataset import RandomDataGenerator
from cbrain.dataset import IntegerDataGenerator
from cbrain.dataset import SequenceDataGenerator
from cbrain.dataset import ProbabilityDataGenerator
from cbrain.function import Square, Fibonacci, Sin


class SquareRandomDataSet(FunctionDataSet, Square, RandomDataGenerator):
    pass


class FibIntegerDataSet(FunctionDataSet, Fibonacci, IntegerDataGenerator):
    pass


class SinRandomDataSet(FunctionDataSet, Sin, RandomDataGenerator):
    pass


class TestDataSet(unittest.TestCase):

    def test_SquareRandomDataSet(self):
        dataset = SquareRandomDataSet()

        i, t = dataset.train_data()
        self.assertEqual(len(i), 1)
        self.assertEqual(len(t), 1)

        i, t = dataset.train_data(0, 1)
        self.assertEqual(len(i), 1)
        self.assertEqual(len(t), 1)

        i, t = dataset.train_data(0, 10)
        self.assertEqual(len(i), 10)
        self.assertEqual(len(t), 10)

    def test_FibIntegerDataSet(self):
        dataset = FibIntegerDataSet()
        i, t = dataset.train_data(5, 8)

        testi = np.array([5]), np.array([6]), np.array([7])
        testt = np.array([5]), np.array([8]), np.array([13])

        self.assertEqual(i, testi)
        self.assertEqual(t, testt)

    def test_SinRandomDataSet(self):
        dataset = SinRandomDataSet()
        i, t = dataset.train_data()

        self.assertEqual(t, np.sin(i))

    @unittest.skip
    def test_MnistDataSet(self):
        dataset = MnistDataSet()
        i, t = dataset.train_data(0, None)


class TestSequenceGenerator(unittest.TestCase):

    def test_SequenceDataGenerator(self):
        seq = (0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
        dg = SequenceDataGenerator(sequence=seq)

        data = dg.get(0, len(seq), 1)

        self.assertTrue(all(x[0]==y for x, y in zip(data.tolist(), list(seq))))


class TestIntegerGenerator(unittest.TestCase):

    def test_IntegerDataGenerator(self):
        dg = IntegerDataGenerator()

        data = dg.get(0, 5, 1)

        self.assertTrue(all(x[0]==y for x, y in zip(data.tolist(), range(5))))


class TestProbabilityGenerator(unittest.TestCase):

    def test_ProbabilityDataGenerator(self):
        pattern = np.array([[1, 1], [3, 3]])
        prob = [0.5, 0.5]
        dg = ProbabilityDataGenerator(pattern, prob)

        data = dg.get(0, 1000, 1)
        a, c = np.unique(data, axis=0, return_counts=True)

        np.testing.assert_equal(pattern, a)
        self.assertEqual(np.round(c[0]/c[1]), 1)


if __name__ == "__main__":
    unittest.main()
