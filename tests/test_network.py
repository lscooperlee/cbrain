
import itertools
import unittest
import numpy as np

from cbrain.network import AbstractNetwork
from cbrain.network import learning_rate
from cbrain.network import weight_decay
from cbrain.network import momentum


class TestAbstractNetwork(unittest.TestCase):

    def test_train(self):
        class TestNetwork(AbstractNetwork):

            def __init__(self):
                super().__init__()
                self.update_theta({'t1':1, 't2':2})

            def train_iter(self, v):
                return {'t1':1, 't2':1}

        network = TestNetwork()

        train_iter = network.train([[]])
        train5 = itertools.islice(train_iter, 5)

        for n, s in enumerate(train5):
            self.assertEqual(s.t1, -n)
            self.assertEqual(s.t2, -n+1)


class TestLearningRateDecor(unittest.TestCase):

    def test_decor(self):

        @learning_rate(10)
        class TestNetwork(AbstractNetwork):

            def __init__(self):
                super().__init__()
                self.update_theta({'t1':1, 't2':2})

            def train_iter(self, v):
                return {'t1':0.1, 't2':0.1}

        network = TestNetwork()

        train_iter = network.train([[]])
        train5 = itertools.islice(train_iter, 5)

        for n, s in enumerate(train5):
            self.assertEqual(s.T['t1'], -n+0.0)
            self.assertEqual(s.T['t2'], -n+1.0)


class TestWeightDecayDecor(unittest.TestCase):

    def test_decor(self):

        @weight_decay(0.5)
        class TestNetwork(AbstractNetwork):

            def __init__(self):
                super().__init__()
                self.update_theta({'t1':1, 't2':3})

            def train_iter(self, v):
                return {'t1':0, 't2':0}

        network = TestNetwork()

        train_iter = network.train([[]])
        train5 = itertools.islice(train_iter, 5)

        t1, t2 = 1, 3
        for n, s in enumerate(train5):
            t1, t2 = t1/2, t2/2
            self.assertEqual(s.t1, t1)
            self.assertEqual(s.t2, t2)


class TestMomentum(unittest.TestCase):

    def test_decor(self):

        @momentum(1)
        class TestNetwork(AbstractNetwork):

            def __init__(self):
                super().__init__()
                self.update_theta({'t1':1, 't2':2})

            def train_iter(self, v):
                return {'t1':1, 't2':1}

        network = TestNetwork()

        train_iter = network.train([[]])
        train4 = itertools.islice(train_iter, 4)

        for n, s in enumerate(train4):
            self.assertEqual(s.t1, -(n//2))
            self.assertEqual(s.t2, (3-n)//2)

if __name__ == "__main__":
    unittest.main()
