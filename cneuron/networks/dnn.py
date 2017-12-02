
import numpy as np
from itertools import zip_longest

from ..functions import Jac, Der, LMS
from .layer import LinearLayer, LayerOutput
from .network import AbstractNetwork


class DynamicNetwork(AbstractNetwork):

    class Cache:

        def __init__(self, maxD):
            self.maxD = maxD
            self.cachedW = {}
            self.cachedB = {}

        def get_weight(self, L, W, T):
            return None
            return self.cachedW.get((L, W.id, T), None)

        def set_weight(self, L, W, T, dW):
            self.cachedW[(L, W.id, T)] = dW

        def get_bias(self, L, B, T):
            return None
            return self.cachedW.get((L, id(B), T), None)

        def set_bias(self, L, B, T, dB):
            self.cachedW[(L, id(B), T)] = dB

    def __init__(self, layers=()):

        self.layers = list(layers)

        for n, l in enumerate(self.layers):
            l.order = n+1

        self.maxD = 0

        self.input_layers = [[] for _ in range(len(self.layers))]

    def load(self, input_size, layer_num=1, W=None, D=0):

        inlayer = self.input_layers[layer_num-1]
        if D < len(inlayer) and inlayer[D]:
            return

        layer = self.layers[layer_num-1]
        input_layer = LinearLayer(input_size)
        input_layer.order = 0

        layer.connectedBy(input_layer, D, W)
        if D >= len(inlayer):
            inlayer.extend([None]*(D+1 - len(inlayer)))

        inlayer[D] = input_layer

        self.maxD = max(self.maxD, D)

    def forward(self, input_values):
        if not isinstance(input_values[0], tuple):
            inputs = [(self.layers[0], 0, input_values)]
        else:
            inputs = input_values

        for l, d, i in inputs:
            self.load(len(i[0]), l.order, D=d)
            self.input_layers[l.order-1][d].A = LayerOutput(i)

        # clear As and Ns so that the results will not
        # be appended to results from last call
        for l in self.layers:
            l.A = LayerOutput(l.size)
            l.N = []

        for T in range(len(inputs[0][2])):
            for l in self.layers:
                l.forward(T)

        ret = self.layers[-1].A._data
        return np.array(ret)

    def __str__(self):
        _str = ["{}".format(l) for l in self.layers]
        return "{}".format(" ".join(_str))

    def connect(self, input_layer_num, output_layer_num, D=0, W=None):
        assert input_layer_num > 0 and output_layer_num > 0

        i = self.layers[input_layer_num-1]
        o = self.layers[output_layer_num-1]

        o.connectedBy(i, D, W)

    def train_iter(self, inputs, outputs, learning_rate=0.1, cost_func=LMS()):
            self.cache = self.Cache(self.maxD)
            #dws, dbs = self._train_iter(inputs, outputs, cost_func)
            dws, dbs = self._train_iter_fwd(inputs, outputs, cost_func)
            self._update_parameters(dws, dbs, learning_rate)

    def _train_iter_fwd(self, input_seq, target_seq, cost_func):

        T = len(input_seq) - 1

        output_seq = self.forward(input_seq)
        Lm = self.layers[-1]

        dF = self._cost_function(target_seq, output_seq, cost_func)

        dws = []
        dbs = []

        for T in range(len(input_seq)):

            for l in self.layers:
                ws = []
                for _, W in l.InLayers:
                    da_dw = self._partialw(Lm, W, T)
                    dF_dw = sum([ t*a for t, a in zip(dF, da_dw)])
                    ws.append(dF_dw)

                if T == len(input_seq) - 1:
                    dws.append(ws)

            for l in self.layers:
                B = l.B
                da_db = self._partialb(Lm, B, T)
                dF_db = sum([ t*a for t, a in zip(dF, da_db)])

                if T == len(input_seq) - 1:
                    dbs.append(dF_db)

        return dws, dbs

    def _train_iter(self, input_seq, target_seq, cost_func):

        T = len(input_seq) - 1

        output_seq = self.forward(input_seq)
        Lm = self.layers[-1]

        dF = self._cost_function(target_seq, output_seq, cost_func)

        dws = []
        dbs = []

        for l in self.layers:
            ws = []
            for _, W in l.InLayers:
                da_dw = self._partialw(Lm, W, T)
                dF_dw = sum([ t*a for t, a in zip(dF, da_dw)])
                ws.append(dF_dw)

            dws.append(ws)

        for l in self.layers:
            B = l.B
            da_db = self._partialb(Lm, B, T)
            dF_db = sum([ t*a for t, a in zip(dF, da_db)])

            dbs.append(dF_db)

        return dws, dbs

    def _update_parameters(self, dws, dbs, learning_rate):

        for n, l in enumerate(self.layers):
            for k in range(len(l.InLayers)):
                l.InLayers[k][1] = l.InLayers[k][1] + (-learning_rate) * dws[n][k]

            l.B = l.B + (-learning_rate) * dbs[n]

    def _partialw(self, L, W, T):

        dNdw = self._direct_partialn_partialw(L, None, 0, W)

        if L.order == 0 or T < 0:
            return dNdw

        for (l, d), w in L.InLayers:
            o = self.cache.get_weight(L, W, T)
            if o is None:
                o = self._partialw(l, W, T-d)
                assert len(o.shape) == 3
                assert o.shape[0] == l.size
                assert o.shape[1:] == W.shape
                self.cache.set_weight(l, W, T, o)

            if W is w:
                dNdw += self._direct_partialn_partialw(L, l, T-d, W)

            dNdw4l = self._multiply3d(w.T, o)

            dNdw += dNdw4l

        dF = Jac(L.F)(L.N[T])

        dAdw = self._multiply3d(dF, dNdw)

        return dAdw

    def _partialb(self, L, B, T):

        dNdb = self._direct_partialn_partialb(L, B, True)

        if L.order == 0 or T < 0:
            return dNdb

        if B is L.B:
            dNdb += self._direct_partialn_partialb(L, B)

        for (l, d), w in L.InLayers:
            o = self.cache.get_bias(L, B, T)
            if o is None:
                o = self._partialb(l, B, T-d)
                self.cache.set_bias(l, B, T, o)

            dNdb4l = self._multiply2d(w.T, o)
            dNdb += dNdb4l

        dF = Jac(L.F)(L.N[T])

        dAdb = self._multiply2d(dF, dNdb)

        return dAdb

    def _cost_function(self, target, output, cost_func):
        return sum((Der(cost_func)(t, o) for t, o in zip(target, output)))

    def _direct_partialn_partialw(self, L, LmInput, T, W):

        dndwlist = []
        for i in range(L.size):
            narray = np.zeros(W.shape)
            if LmInput:
                AmInput = np.array(LmInput.A._data[T])
                narray[:, i] = AmInput

            dndwlist.append(narray)

        return np.array(dndwlist)

    def _direct_partialn_partialb(self, L, B, init=False):
        dndblist = []
        for i in range(L.size):
            narray = np.zeros(B.shape)
            if not init:
                narray[i] = 1
            dndblist.append(narray)

        return np.array(dndblist)

    def _multiply_2d_3d(self, vector2d, vector3d):
        assert vector2d.shape[1] == len(vector3d)
        tmp = []
        for v in vector2d:
            t = sum((x*y for x, y in zip(v, vector3d)))
            tmp.append(t)
        return np.array(tmp)

    def _multiply_2d_2d(self, _vector2d, vector2d):
        assert _vector2d.shape[1] == len(vector2d)
        tmp = []
        for v in _vector2d:
            t = sum((x*y for x, y in zip(v, vector2d)))
            tmp.append(t)
        return np.array(tmp)

    def _multiply3d(self, vector, vector3d):
        assert len(vector3d.shape) == 3
        return self._multiply_2d_3d(vector, vector3d)

    def _multiply2d(self, vector, vector2d):
        assert len(vector2d.shape) == 2
        return self._multiply_2d_2d(vector, vector2d)


class RadialBasisNetwork:

    def __init__(self, layers=(), Ws=None, Bs=None):

        self.layers = list(layers)

        # for l, w in zip_longest(self.layers, (Ws,) if Ws is None else Ws):
        #    pass

    def forward(self, input_value):

        for l in self.layers:
            input_value = l.forward(input_value)

        return input_value
