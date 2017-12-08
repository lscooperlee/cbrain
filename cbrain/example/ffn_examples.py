
import matplotlib.pyplot as plt
import numpy as np
import itertools

from ..network.ffn import FFN
from ..network.network import learning_rate
from ..function import Logsig, Square, Sin
from ..dataset import FunctionDataSet
from ..dataset import RandomDataGenerator


class SquareRandomDataSet(FunctionDataSet, Square, RandomDataGenerator):
    pass


class SinRandomDataSet(FunctionDataSet, Sin, RandomDataGenerator):
    pass

@learning_rate()
class SimpleFFN(FFN):
    pass

def FFN_on_SquareRandomDataSet(n=10000):

    net = SimpleFFN((1, 2, 1))

    d = SquareRandomDataSet()

    test_inputs, test_outputs = d.train_data(0, 10000)

    train_iter = net.train(zip(test_inputs, test_outputs))
    train_n = itertools.islice(train_iter, n)

    for s in train_n:
        pass

    eval_data, eval_out = d.test_data(0, 100)
    eval_perform = [net.forward(d) for d in eval_data]

    plt.plot(eval_data, eval_out, "*")
    plt.plot(eval_data, eval_perform, "r.")
    plt.show()


def DNN_as_FFN_on_SinRandomDataSet():
    l1 = Layer(2, Logsig())
    l3 = Layer(8, Logsig())
    l2 = Layer(1, Logsig())

    net = Network((l1, l3, l2))
    net.connect(1, 2, D=0)
    net.connect(2, 3, D=0)

    d = SinRandomDataSet(0, 6.28)

    _test_inputs, test_outputs = d.train_data(0, 1000000)
    test_inputs = [np.array([x]) for x in _test_inputs]

    net.train(test_inputs, test_outputs)

    eval_data, eval_out = d.test_data(0, 100)
    eval_perform = net.forward(eval_data)

    plt.plot(eval_data, eval_out, "*")
    plt.plot(eval_data, eval_perform, "r.")
    plt.show()
