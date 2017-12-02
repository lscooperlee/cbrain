import matplotlib.pyplot as plt
import numpy as np
import itertools

from ..networks.rbm import RBM
from ..networks.network import learning_rate
from ..networks.network import weight_decay
from ..networks.network import momentum
from ..dataset.data_generator import ProbabilityDataGenerator

@learning_rate()
#@weight_decay()
#@momentum()
class SimpleBernoulliRBM(RBM):
    pass


def RBM_on_GenerateData(times=10000):
    network = SimpleBernoulliRBM((2, 1))
    pattern = [[0, 0], [0, 1], [1, 0], [1, 1]]
    dg = ProbabilityDataGenerator(pattern = pattern,
                                  prob = [0.1, 0.4, 0.4, 0.1])

    data = dg.get(0, 1000, 1)

    train_iter = network.train(data)
    train = itertools.islice(train_iter, times)
    for s in train:
#        print(s.theta)
        pass

    generated_data = network.generate(1000)

    p, c = np.unique(data, return_counts=True, axis=0)
    plt.bar([p-0.2 for p in range(len(pattern))], c, width=0.4)

    p, c = np.unique(generated_data, return_counts=True, axis=0)
    print(p, c)
    plt.bar([p+0.2 for p in range(len(pattern))], c, width=0.4)

    plt.show()

