
import abc

from .data_generator import DataGenerator
from ..function import Function


class DataSet(abc.ABC):

    @abc.abstractmethod
    def itrain_data(self):
        """
            return iterator of training data pair
        """

    @abc.abstractmethod
    def itest_data(self):
        """
            return iterator of testing data pair
        """

    def __data_handler(self, iterator, start=0, stop=1, step=1):
        if stop is not None and stop - start == step:
            return next(iterator(start, stop, step))
        else:
            it = iterator(start, stop, step)
            return tuple(zip(*it))

    def train_data(self, start=0, stop=1, step=1):
        return self.__data_handler(self.itrain_data, start, stop, step)

    def test_data(self, start=0, stop=1, step=1):
        return self.__data_handler(self.itrain_data, start, stop, step)


class FunctionDataSet(DataSet, Function, DataGenerator):

    def _data(self, start, stop, step):

        D = self.iget(start, stop, step)
        for d in D:
            O = self(d)
            yield (d, O)

    def itrain_data(self, start=0, stop=None, step=1):
        return self._data(start, stop, step)

    def itest_data(self, start=0, stop=None, step=1):
        return self._data(start, stop, step)


class SerialFunctionDataSet(FunctionDataSet):

    def __init__(self):
        self.__data = []

    def _data(self, start, stop, step):

        D = self.iget(start, stop, step)

        for d in D:
            self.__data.append(d)
            O = self(self._data)
            yield (d, O)
