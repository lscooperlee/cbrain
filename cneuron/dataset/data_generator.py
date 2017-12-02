
import abc
import itertools
import inspect

import numpy as np


class DataGenerator(abc.ABC):

    def __init__(self, shape=(1,)):
        self.shape = shape

    @abc.abstractmethod
    def iget(self, start=0, stop=None, step=1):
        pass

    def get(self, start=0, stop=1, step=1):
        if stop is None or stop - start != step:
            if not inspect.signature(self.iget).parameters:
                it = itertools.islice(self.iget(), start, stop, step)
            else:
                it = self.iget(start, stop, step)
            return np.array(list(it))
        else:
            if stop < start:
                stop = start + step
            return next(self.iget(start, stop, step))


class RandomDataGenerator(DataGenerator):

    def __init__(self, low=0, high=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low = low
        self.high = high

    def iget(self, start=0, stop=None, step=1):
        for _ in range(start, stop, step) if stop else itertools.count():
            random = np.random.random_sample(self.shape)
            yield (self.high - self.low)*random + self.low


class IntegerDataGenerator(DataGenerator):

    def iget(self, start=0, stop=None, step=1):
        import sys
        for i in range(start, stop or sys.maxint, step):
            yield np.array([i])


class SequenceDataGenerator(DataGenerator):

    def __init__(self, sequence, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence = tuple(sequence)

    def iget(self, start=0, stop=None, step=1):

        for i in range(start, stop or len(self.sequence), step):
            try:
                yield np.array([self.sequence[i]])
            except:
                raise StopIteration


class ProbabilityDataGenerator(DataGenerator):

    def __init__(self, pattern, prob=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern = pattern
        self.prob = prob

    def iget(self, start=0, stop=None, step=1):
        for _ in range(start, stop, step) if stop else itertools.count():
            n = np.random.choice(len(self.pattern), p=self.prob)
            yield self.pattern[n]
