
import abc
import itertools
import numpy as np
from collections import defaultdict


class AbstractNetwork(abc.ABC):

    def __init__(self):
        self.T = {}

    def __getattr__(self, theta):
        try:
            return self.T[theta]
        except:
            raise AttributeError

    def update_theta(self, key, value=None):
        if value is None:
            self.T.update(key)
        else:
            self.T[key] = value

    @abc.abstractmethod
    def train_iter(self, d):
        pass

    def _train(self, ds):
        for d in itertools.cycle(ds):

            dtheta = self.train_iter(d)

            for k, v in dtheta.items():
                self.T[k] = self.T[k] - v

            yield self


    def train(self, ds):
        return self._train(ds)


def learning_rate(rate = 0.1):

    def decor(cls):
        class wrapper(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.learning_rate = rate

            def train_iter(self, d):
                origin_delta = super().train_iter(d)

                for k, v in origin_delta.items():
                    origin_delta[k] = self.learning_rate * v

                return origin_delta

        return wrapper

    return decor


def weight_decay(decay = 0.01):

    def decor(cls):
        class wrapper(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def train_iter(self, d):
                learning_rate = getattr(self, 'learning_rate', 1)
                origin_delta = super().train_iter(d)

                for k, v in origin_delta.items():
                    theta = self.T[k]
                    origin_delta[k] = v + learning_rate * decay * theta

                return origin_delta

        return wrapper

    return decor


def momentum(alpha = 0.75):

    def decor(cls):
        class wrapper(cls):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.last_delta_theta = defaultdict(int)

            def train_iter(self, d):
                origin_delta = super().train_iter(d)

                for k, v in origin_delta.items():
                    last_delta_theta = self.last_delta_theta[k]
                    origin_delta[k] = v - alpha * last_delta_theta
                    self.last_delta_theta[k] = origin_delta[k]

                return origin_delta

        return wrapper

    return decor
