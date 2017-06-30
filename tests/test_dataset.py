import unittest
import numpy as np

from cneuron.dataset import FunctionDataSet
from cneuron.dataset import MnistDataSet
from cneuron.dataset import RandomDataGenerator, IntegerDataGenerator
from cneuron.dataset import SequenceDataGenerator
from cneuron.functions import Square, Fibonacci, Sin


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
        i, t = dataset.train_data()


class TestDataGenerator(unittest.TestCase):

    def test_SequenceDataGenerator(self):
        seq = (0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
        dg = SequenceDataGenerator(sequence=seq)

        data = dg.get(0, len(seq), 1)

        self.assertTrue(all(x[0]==y for x, y in zip(data.tolist(), list(seq))))

    def test_IntegerDataGenerator(self):
        dg = IntegerDataGenerator()

        data = dg.get(0, 5, 1)

        self.assertTrue(all(x[0]==y
                            for x, y in zip(data.tolist(), range(5))))

if __name__ == "__main__":
    unittest.main()
