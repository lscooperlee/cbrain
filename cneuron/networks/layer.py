
import abc
import numpy as np
import itertools
import functools
import collections

from ..functions import Jac, Der, Line, LMS


class Weight(np.ndarray):

    def __new__(cls, input_array, input_layer, delay, owned_layer):
        obj = np.asarray(input_array).view(cls)
        obj.layer = input_layer
        obj.output = owned_layer
        obj.D = delay
        obj.id = hash((input_layer, delay, owned_layer))

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.layer = getattr(obj, 'layer', None)
        self.output = getattr(obj, 'output', None)
        self.D = getattr(obj, 'D', None)
        self.id = getattr(obj, 'id', None)


class Bias(np.ndarray):

    def __new__(cls, input_array, input_layer, delay):
        obj = np.asarray(input_array).view(cls)
        obj.layer = input_layer

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.layer = getattr(obj, 'layer', None)


class LayerOutput:
    def __init__(self, arg):
        if isinstance(arg, int):
            self._size = arg
            self._data = []
        else:
            self._data=list(arg)
            self._size=len(self._data[0])

    def __getitem__(self, i):
        if i < 0:
            return np.zeros(self._size)
        return self._data[i]

    def append(self, d):
        self._data.append(d)

    def __len__(self):
        return self._size


class Layer(abc.ABC):

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def connectedBy(self):
        pass


class SimpleLayer(Layer):

    def __init__(self, size, func=Line(1, 0), W=None, B=None):

        self.size = size
        self.B = np.random.rand(self.size) if B is None else B
        self.W = W
        self.F = func

        self.N = np.ones(self.size)
        # self.A = self.N

    def connectedBy(self, layer, W=None):
        if self.W is None:
            self.W = np.random.rand(layer.size, self.size) if W is None else W

    def __str__(self):
        if self.order < 0:
            _str = "[{0[0]} X {0[1]}]".format(self.W.shape)
        else:
            _str = "[{0[0]} X {0[1]} @{1}]".format(self.W.shape, self.order)

        return _str


class DynamicLayer(Layer):

    def __init__(self, size=0, func=Line(1, 0), B=None):

        self.size = size
        self.F = func
        self.A = LayerOutput(self.size)
        self.N = []
        self.InLayers = []

        self.B = np.random.rand(self.size) if B is None else B

    def connectedBy(self, layer, D=0, W=None):
        if W is None:
            W = np.random.rand(layer.size, self.size)

        inLayer = self.InLayer(layer, D)
        self.InLayers.append([inLayer, Weight(W, layer, D, self)])

    def __str__(self):
        if hasattr(self, "order"):
            _str = "[L{0}: {1}]".format(self.order, self.size)
        else:
            _str = "[L: {1}]".format(self.size)

        return _str


class FeedForwardLayer(SimpleLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_layer):
        self.N = np.transpose(self.W).dot(input_layer) + self.B
        self.A = self.F(self.N)
        return self.A


class LinearLayer(DynamicLayer):

    InLayer = collections.namedtuple("InLayer", ("L", "D"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, T):

        N = sum(( l.A[T-d]@w for (l, d), w in self.InLayers)) + self.B
        self.N.append(N)

        A = self.F(N)
        self.A.append(A)

        return A


class RadialLayer(SimpleLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_value):
        lstA = []
        for Wi, Bi in zip(self.W.T, self.B):
            ret = np.linalg.norm(input_value - Wi)*Bi
            lstA.append(ret)


        #lstA = [np.linalg.norm(input_value - Wi)*Bi
        #            for Wi, Bi in zip(self.W.T, self.B)]

        self.N = np.array(lstA)
        self.A = self.F(self.N)
        return self.A
