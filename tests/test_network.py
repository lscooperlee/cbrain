
import itertools
import unittest
import numpy as np

from cneuron.networks import AbstractNetwork
from cneuron.networks import learning_rate
from cneuron.networks import weight_decay
from cneuron.networks import momentum


class TestAbstractNetwork(unittest.TestCase):

    def test_train(self):
        class TestNetwork(AbstractNetwork):

            def __init__(self):
                super().__init__()
                self.T['t1'] = 1
                self.T['t2'] = 2

            def train_iter(self, v):
                return {'t1':1, 't2':1}

        network = TestNetwork()

        train_iter = network.train([[]])
        train5 = itertools.islice(train_iter, 5)

        for n, s in enumerate(train5):
            self.assertEqual(s.T['t1'], -n)
            self.assertEqual(s.T['t2'], -n+1)


class TestLearningRateDecor(unittest.TestCase):

    def test_decor(self):

        @learning_rate(10)
        class TestNetwork(AbstractNetwork):

            def __init__(self):
                super().__init__()
                self.T['t1'] = 1
                self.T['t2'] = 2

            def train_iter(self, v):
                return {'t1':0.1, 't2':0.1}

        network = TestNetwork()

        train_iter = network.train([[]])
        train5 = itertools.islice(train_iter, 5)

        for n, s in enumerate(train5):
            self.assertEqual(s.T['t1'], -n+0.0)
            self.assertEqual(s.T['t2'], -n+1.0)


# class TestWeightDecayDecor(unittest.TestCase):

    # def test_decor(self):

        # @weight_decay(2)
        # class TestNetwork(AbstractNetwork):

            # def __init__(self):
                # self.t1 = 0
                # self.t2 = 0

                # super().__init__(self.t1, self.t2)

            # def train_iter(self, v):
                # return [1, 1]

        # network = TestNetwork()

        # train_iter = network.train([[]])
        # train5 = itertools.islice(train_iter, 5)

        # origin = network.theta
        # for n, s in enumerate(train5):
            # self.assertEqual(s.theta, [-1+n%2, -1+n%2])


# class TestMomentum(unittest.TestCase):

    # def test_decor(self):

        # @momentum(2)
        # class TestNetwork(AbstractNetwork):

            # def __init__(self):
                # self.t1 = 0
                # self.t2 = 0

                # super().__init__(self.t1, self.t2)

            # def train_iter(self, v):
                # return [1, 2]

        # network = TestNetwork()

        # train_iter = network.train([[]])
        # train5 = itertools.islice(train_iter, 5)

        # origin = network.theta
        # for n, s in enumerate(train5, 1):
            # self.assertEqual(s.theta, [-n, -n*2])

if __name__ == "__main__":
    unittest.main()
