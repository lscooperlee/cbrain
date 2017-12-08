
import matplotlib.pyplot as plt
import numpy as np

from ..network.neuron import DynamicNetwork
from ..network.layer import LinearLayer
from ..function import Logsig, Square, Sin
from ..dataset import FunctionDataSet
from ..dataset import RandomDataGenerator


class SquareRandomDataSet(FunctionDataSet, Square, RandomDataGenerator):
    pass


class SinRandomDataSet(FunctionDataSet, Sin, RandomDataGenerator):
    pass


def DNN_as_FFN_on_SquareRandomDataSet():
    l1 = LinearLayer(2, Logsig())
    l2 = LinearLayer(1, Logsig())

    net = DynamicNetwork((l1, l2))
    net.connect(1, 2, D=0)

    d = SquareRandomDataSet()

    _test_inputs, test_outputs = d.train_data(0, 10000)
    test_inputs = [np.array([x]) for x in _test_inputs]

    net.train(test_inputs, test_outputs)

    eval_data, eval_out = d.test_data(0, 100)
    eval_perform = net.forward(eval_data)

    plt.plot(eval_data, eval_out, "*")
    plt.plot(eval_data, eval_perform, "r.")
    plt.show()


def DNN_as_FFN_on_SinRandomDataSet():
    l1 = LinearLayer(2, Logsig())
    l3 = LinearLayer(8, Logsig())
    l2 = LinearLayer(1, Logsig())

    net = DynamicNetwork((l1, l3, l2))
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
