
import abc
import itertools
import numpy as np


class AbstractNetwork(abc.ABC):

    def __init__(self):
        self.T = {}

    @abc.abstractmethod
    def train_iter(self, v):
        pass

    def train_data(self, data):
        for v in data:
            dtheta = self.train_iter(v)

            for k, v in dtheta.items():
                self.T[k] = self.T[k] - v

            yield self

    def train(self, data):
        while True:
            np.random.shuffle(data)
            yield from self.train_data(data)


def learning_rate(rate = 0.1):

    def decor(cls):
        class wrapper(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.learning_rate = rate

            def train_iter(self, v):
                origin_delta = super().train_iter(v)

                for k, v in origin_delta.items():
                    origin_delta[k] = self.learning_rate * v

                return origin_delta

        return wrapper

    return decor


def weight_decay(decay = 0.9):

    def decor(cls):
        class wrapper(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def train_iter(self, v):
                learning_rate = getattr(self, 'learning_rate', 1)
                origin_delta = super().train_iter(v)
                return [o + decay*learning_rate*t
                        for t, o in zip(self.theta, origin_delta)]

        return wrapper

    return decor


def momentum(alpha = 0.5):

    def decor(cls):
        class wrapper(cls):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.last_delta_theta = [0] * len(self.theta)

            def train_iter(self, v):
                origin_delta = super().train_iter(v)
                return [o + alpha*t
                        for t, o in zip(self.last_delta_theta, origin_delta)]

        return wrapper

    return decor
