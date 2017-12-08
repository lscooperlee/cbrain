import matplotlib.pyplot as plt
import numpy as np
import itertools

from ..network.rbm import RBM
from ..network.network import learning_rate
from ..network.network import weight_decay
from ..network.network import momentum
from ..dataset.data_generator import ProbabilityDataGenerator
from ..dataset.mnist_dataset import MnistDataSet

@learning_rate()
@weight_decay()
@momentum()
class SimpleBernoulliRBM(RBM):
    pass


def RBM_on_GenerateData(times=10000):
    network = SimpleBernoulliRBM((2, 1))
    pattern = [[0, 0], [0, 1], [1, 0], [1, 1]]
    dg = ProbabilityDataGenerator(pattern = pattern,
                                  prob = [0.1, 0.4, 0.4, 0.1])
                                  #prob = [0.0, 0.8, 0.2, 0.0])
                                  #prob = [0.1, 0.7, 0.2, 0.0])

    print(network.ph([0]))
    print(network.ph([1]))

    data = dg.get(0, 1000, 1)

    train_iter = network.train(data)
    train = itertools.islice(train_iter, times)
    for s in train:
        pass

    generated_data = network.generate(1000)

    p, c = np.unique(data, return_counts=True, axis=0)
    plt.bar([p-0.2 for p in range(len(pattern))], c, width=0.4)

    p, c = np.unique(generated_data, return_counts=True, axis=0)
    plt.bar([p+0.2 for p in range(len(pattern))], c, width=0.4)

    print(network.ph([0]))
    print(network.ph([1]))

    plt.show()


def RBM_on_Mnist(times=1000):
    network = SimpleBernoulliRBM((784, 500))

    print(network)

    dataset = MnistDataSet()
    i = dataset.itrain_data(0, None, 1)
    traind = (_[0] for _ in i)

    train_iter = network.train(traind)
    train = itertools.islice(train_iter, times)
    for s in train:
        pass

    print("done")
    h, v = network.generate()

    plt.imshow(h.reshape(28, 28))
    plt.show()
