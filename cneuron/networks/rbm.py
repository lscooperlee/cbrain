import numpy as np
import itertools

from ..functions import Logsig
from .network import AbstractNetwork


class RBM(AbstractNetwork):

    def __init__(self, size):
        super().__init__()
        self.T['W'] = np.random.random(size)
        self.T['B'] = np.random.random(size[0])
        self.T['C'] = np.random.random(size[1])

        self.size = size

        self.S = Logsig()


    def CD(self, V, k=1):
        Vk = self.sample(V, k)

        dW = np.outer(Vk, self.S(self._alpha(Vk))) \
            - np.outer(V, self.S(self._alpha(V)))
        dB = Vk - V
        dC = self.S(self._alpha(Vk)) - self.S(self._alpha(V))

        return {"W":dW, "B":dB, "C":dC}

    def _alpha(self, V):
        return self.T['W'].T @ V + self.T['C']

    def _beta(self, H):
        return self.T['W'] @ H + self.T['B']

    def sample(self, V, k=1):
        for _ in range(k):
            H = self.S(self._alpha(V)) > np.random.random(len(self.T['C']))
            V = self.S(self._beta(H)) > np.random.random(len(self.T['B']))
        return V

    def generate(self, size=1):
        ALL_V = np.array(list(itertools.product(range(2), repeat=self.size[0])))
        ALL_H = np.array(list(itertools.product(range(2), repeat=self.size[1])))

        def etoE(V, H):
            E = V@self.T['W']@H + V@self.T['B'] + H@self.T['C']
            return np.exp(E)

        pv = np.array([sum([etoE(v, h) for h in ALL_H]) for v in ALL_V])
        idx = np.random.choice(len(ALL_V), p=pv/pv.sum(), size=size)

        return [ALL_V[i] for i in idx]


    def train_iter(self, v):
        return self.CD(v)
