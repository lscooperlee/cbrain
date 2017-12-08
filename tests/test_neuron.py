
import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from cbrain.network import LinearLayer, RadialLayer, FeedForwardLayer
from cbrain.network import DynamicNetwork, RadialBasisNetwork
from cbrain.dataset import IntegerDataGenerator, RandomDataGenerator
from cbrain.dataset import SequenceDataGenerator
from cbrain.dataset import FunctionDataSet
from cbrain.function import LMS, Logsig, Square, Line, CrossEntropy
from cbrain.function import Gaussian


class TestDynamicNetworks(unittest.TestCase):

    def test_DNN_forward(self):
        l1 = LinearLayer(2, B=np.ones((2, )))
        l2 = LinearLayer(1, B=np.ones((1,)))

        net = DynamicNetwork((l1, l2))
        net.connect(1, 2, D=0, W=np.array([[2], [1]]))

        dg = IntegerDataGenerator()
        idata = dg.get(1, 3, 1)

        net.load(len(idata[0]), 1, W=np.array([[3, 1]]), D=0)
        odata = net.forward(idata)

        assert_array_equal(odata, np.array([[11], [18]]))

    def test_DNN_examples(self):
        """see https://au.mathworks.com/help/nnet/ug/
               how-dynamic-neural-networks-work.html
        """

        class LineSequenceDataSet(FunctionDataSet, Line, SequenceDataGenerator):
            pass

        seq = (0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
        dataset = LineSequenceDataSet(sequence=seq, T=2)
        data, tdata = dataset.train_data(0, len(seq))

        # example 1
        def example1(data):
            l1 = LinearLayer(1, B=np.zeros((1,)))

            net1 = DynamicNetwork((l1, ))
            net1.load(1, 1, W=np.array([[2]]), D=0)
            return net1.forward(data)

        assert_array_equal(tdata, example1(data))

        tdata2 = np.array([[0], [0], [1], [2], [2], [2], [1],
                          [0], [0], [0], [0], [0]])

        # example 2
        def example2(data):
            l2 = LinearLayer(1, B=np.zeros((1,)))
            net2 = DynamicNetwork((l2, ))
            net2.load(1, 1, W=np.array([[1]]), D=1)
            net2.load(1, 1, W=np.array([[1]]), D=0)

            inputdata = [(l2, 0, data), (l2, 1, data)]
            return net2.forward(inputdata)

        assert_array_equal(tdata2, example2(data))

        tdata3 = np.array([[0], [0], [1], [1.5], [1.75], [1.875], [0.9375],
                           [0.46875], [0.234375], [0.1171875],
                           [0.05859375], [0.02929688]])

        # example 3
        def example3(data):
            l3 = LinearLayer(1, B=np.zeros((1,)))
            net2 = DynamicNetwork((l3, ))
            net2.load(1, 1, W=np.array([[1]]), D=0)
            net2.connect(1, 1, W=np.array([[0.5]]), D=1)

            return net2.forward(data)

        assert_almost_equal(tdata3, example3(data))

    def test_DNN_load(self):
        l1 = LinearLayer(2, B=np.ones((2, )))
        l2 = LinearLayer(1, B=np.ones((1,)))

        net = DynamicNetwork((l1, l2))
        net.connect(1, 2, D=0, W=np.array([[2], [1]]))

        loadedW = np.array([[3, 1]])
        net.load(1, 1, W=loadedW, D=1)
        net.load(1, 1, W=loadedW, D=0)

        assert_array_equal(net.layers[0].InLayers[0][1], loadedW)

    def test_DNN_train_iter(self):
        l1 = LinearLayer(2, B=np.ones((2, )), func=Logsig())
        l2 = LinearLayer(1, B=np.ones((1,)), func=Logsig())

        net = DynamicNetwork((l1, l2))
        net.connect(1, 2, D=0, W=np.array([[2], [1]]))

        dg = IntegerDataGenerator()
        idata = dg.get(-1, 2, 1)
        tdata = dg.get(1, -2, -1)

        net.train_iter(idata, tdata, cost_func=LMS())

    def test_DNN_train(self):

        class SquareRandomDataSet(FunctionDataSet, Square, RandomDataGenerator):
            pass

        l1 = LinearLayer(3, Logsig())
        l2 = LinearLayer(2, Logsig())

        net = DynamicNetwork((l1, l2))
        net.connect(1, 2, D=0)

        d = SquareRandomDataSet(shape=(1,))

        test_inputs, test_outputs = d.train_data(0, 100)

        _eval_data, _eval_out = d.test_data(0, 10)
        eval_data = [np.array([x]) for x in _eval_data]
        eval_out = [np.array([x]) for x in _eval_out]

        net.train(eval_data, eval_out, cost_func=LMS())
        net.train(eval_data, eval_out, cost_func=CrossEntropy())


class TestRadialNetwork(unittest.TestCase):

    def test_RadialNetwork_forward(self):

        def example1():
            '''
            P16-4
            '''
            W1 = np.array([[-1, 1]])
            B1 = np.array([2, 2])

            W2 = np.array([[1], [1]])
            B2 = np.array([0])

            l1 = RadialLayer(2, W=W1, B=B1, func=Gaussian())
            l2 = FeedForwardLayer(1, W=W2, B=B2)

            net = RadialBasisNetwork((l1, l2))

            assert_almost_equal(net.forward(np.array([-1])), 1)
            assert_almost_equal(net.forward(np.array([1])), 1)
            assert_almost_equal(net.forward(np.array([0])), 0.0366313)
            assert_almost_equal(net.forward(np.array([-2])), 0.0183156)
            assert_almost_equal(net.forward(np.array([2])), 0.0183156)


        def example2():
            '''
            P16-6
            '''

            W1 = np.array([[-1, 1], [1, -1]])
            B1 = np.array([1, 1])

            W2 = np.array([[2], [2]])
            B2 = np.array([-1])

            l1 = RadialLayer(2, W=W1, B=B1, func=Gaussian())
            l2 = FeedForwardLayer(1, W=W2, B=B2)

            net = RadialBasisNetwork((l1, l2))

            assert_almost_equal(net.forward(np.array([-1, 1])), 1.0006709)
            assert_almost_equal(net.forward(np.array([1, -1])), 1.0006709)

            assert_almost_equal(net.forward(np.array([-1, -1])), -0.9267374)
            assert_almost_equal(net.forward(np.array([1, 1])), -0.9267374)


        example1()
        example2()


class TestRadialLayer(unittest.TestCase):

    def test_RBN_forward(self):
        W1 = np.array([[-1, 1], [1, -1]])
        B1 = np.array([1, 1])
        l1 = RadialLayer(2, W=W1, B=B1, func=Gaussian())
        input_value = np.array([-1, -1])

        ret = l1.forward(input_value)
        #print(ret)


if __name__ == "__main__":
    unittest.main()
