import numpy as np
from itertools import product

from ..function import Logsig
from ..function import Jac, Der
from ..function import LMS
from .network import AbstractNetwork


class FFN(AbstractNetwork):

    def __init__(self, size, F=Logsig(), Fcost = LMS()):
        super().__init__()
        self.size = size
        self.F = F
        self.Fcost = Fcost
        self.A = [0] * len(size)
        self.N = [0] * len(size)

        for m, (I, J) in enumerate(zip(size[:-1], size[1:]), 1):
            l = m - 1

            self.update_theta("W%d%d"%(l, m), np.random.random((I, J)))
            self.update_theta("B%d"%(m), np.random.random(J))


    def forward(self, v):

        self.N[0] = v
        self.A[0] = v

        for m in range(1, len(self.size)):
            l = m - 1

            W = self.T["W%d%d"%(l, m)]
            B = self.T["B%d"%(m)]

            self.N[m] = self.N[l]@W + B
            self.A[m] = self.F(self.N[m])

        return self.A[-1]


    def train_iter(self, training_data):
        v, t = training_data
        o = self.forward(v)

        S = Jac(self.F)(self.N[-1]) @ Der(self.Fcost)(t, o)

        dT = {}
        for m in reversed(range(1, len(self.size))):
            l = m-1

            W = self.T["W%d%d"%(l, m)]
            B = self.T["B%d"%(m)]

            dW = np.outer(self.A[l], S)
            dB = S

            dT["W%d%d"%(l, m)] = dW
            dT["B%d"%(m)] = dB

            S = Jac(self.F)(self.N[l]) @ W @ S

        return dT
