import numpy as np
from itertools import product

from ..function import Logsig
from .network import AbstractNetwork

class BM:

    def __init__(self, size):

        self.ALL_V = np.array(list(product(range(2), repeat=size[0])))
        self.ALL_H = np.array(list(product(range(2), repeat=size[1])))
        self.ALL = list(product(self.ALL_V, self.ALL_H))

    def Energy(self, V, H):
        return V@self.W@H + V@self.B + H@self.C

    def phat(self, V, H):
        return np.exp(self.Energy(V, H))

    def p(self, V, H):
        return self.phat(V, H)/self.Z

    def pv(self, V):
        return sum(self.p(V, h) for h in self.ALL_H)

    def ph(self, H):
        return sum(self.p(v, H) for v in self.ALL_V)

    @property
    def Z(self):
        return sum(self.phat(v, h) for v, h in self.ALL)

    def generate(self, size=1):

        pv = np.array([sum([self.phat(v, h) for h in self.ALL_H])
                       for v in self.ALL_V])
        idx = np.random.choice(len(self.ALL_V), p=pv/pv.sum(), size=size)

        return [self.ALL_V[i] for i in idx]


class RBM(AbstractNetwork):

    def __init__(self, size):
        super().__init__()

        self.update_theta('W', 0.1*np.random.random(size)-0.05)
        self.update_theta('B', 0.1*np.random.random(size[0])-0.05)
        self.update_theta('C', 0.1*np.random.random(size[1])-0.05)

        self.S = Logsig()


    def CD(self, V, k=1):
        Vk, _ = self.sample(V, k)

        dW = np.outer(Vk, self.S(self._alpha(Vk))) \
            - np.outer(V, self.S(self._alpha(V)))
        dB = Vk - V
        dC = self.S(self._alpha(Vk)) - self.S(self._alpha(V))

        return {"W":dW, "B":dB, "C":dC}

    def _alpha(self, V):
        return self.W.T @ V + self.C

    def _beta(self, H):
        return self.W @ H + self.B

    def sample(self, V, k=1):
        for _ in range(k):
            H = self.S(self._alpha(V)) > np.random.random(len(self.C))
            V = self.S(self._beta(H)) > np.random.random(len(self.B))
        return V, H

    def train_iter(self, v):
        return self.CD(v)

    def forward(self, V):
            ba = self.S(self._alpha(V)) > np.random.random(len(self.C))
            return ba.astype(int)

    def generate(self):
        V = np.random.random(len(self.B)) > 0.5
        return self.sample(V, 100)
