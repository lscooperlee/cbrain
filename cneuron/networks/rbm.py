import numpy as np
import itertools


class RBM:

    def __init__(self, size_v, size_h, W=None, B=None, C=None):
        self.size_v = size_v
        self.size_h = size_h

        if W is None:
            self.W = np.random.random((size_v, size_h))
        else:
            self.W = W

        if B is None:
            self.B = np.random.random(size_v)
        else:
            self.B = B

        if C is None:
            self.C = np.random.random(size_h)
        else:
            self.C = C


    def forward(self, v):
        pass


    def E(self, v, h):
        positive_e = v @ self.W @ h + v @ self.B + h @ self.C
        return -positive_e

    def eTomE(self, v, h):
        negative_e = -self.E(v, h)
        e = np.exp(negative_e)
        return e

    def Z(self):
        all_v = itertools.product([0, 1], repeat=self.size_v)
        all_h = itertools.product([0, 1], repeat=self.size_h)

        all_vh = itertools.product((all_v), (all_h))

        return sum(self.eTomE(v, h) for v, h in all_vh)

    def P(self, v, h):
        return self.eTomE(v, h) / self.Z()
