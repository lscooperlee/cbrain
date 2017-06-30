
import abc
import numpy as np


def Der(func):
    return func.derivative()


def Jac(func):
    return func.jacobian()


class Function(abc.ABC):

    @abc.abstractmethod
    def derivative(self):
        '''
            return derivative of the function
        '''

    @abc.abstractmethod
    def call(self, x):
        return x

    def __call__(self, *x):
        mro = self.__class__.__mro__
        for f in mro:
            if "call" in vars(f):
                x = f.call(self, *x)
                if not isinstance(x, tuple):
                    x = (x,)

        if len(x) is 1:
            return x[0]


class LinearIndepedentFunction(Function):

    def jacobian(self):
        def jac(*args, **kwargs):
            return np.diag(self.derivative()(*args, **kwargs))
        return jac


class Square(LinearIndepedentFunction):

    def call(self, x):
        return np.square(x)

    def derivative(self):
        def d(x):
            return x*2
        return d


class Line(LinearIndepedentFunction):

    def __init__(self, T=1, B=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = T
        self.B = B

    def call(self, x):
        return self.T*x + self.B

    def derivative(self):
        def d(x):
            return self.T * np.ones(x.shape)
        return d


class Logsig(LinearIndepedentFunction):
    def call(self, n):
        return 1/(1 + np.exp(-n))

    def derivative(self):
        def d(n):
            return (1 - self(n))*self(n)
        return d


class Minus(LinearIndepedentFunction):

    def call(self, t, o):
        return t - o

    def derivative(self):
        pass


class Dot(LinearIndepedentFunction):

    def call(self, x):
        return np.dot(x, x)

    def derivative(self):
        pass


class LMS(Minus, Dot):

    def derivative(self):
        def d(t, o):
            return -2*(t - o)
        return d


class CrossEntropy(LinearIndepedentFunction):

    def call(self, t, o):
        return -sum((_t*np.log(_o) for _t, _o in zip(t, o)))

    def derivative(self):
        def d(t, o):
            return -t/o
        return d


class Tansig(LinearIndepedentFunction):
    def call(self, n):
        return (np.exp(n) - np.exp(-n))/(np.exp(n) + np.exp(-n))

    def derivative(self):
        def d(n):
            return 1 - self(n)*self(n)
        return d


class ReLU(LinearIndepedentFunction):
    def call(self, n):
        return np.maximum(n, 0)

    def derivative(self):
        def d(n):
            return 1 * (n > 0)
        return d


class Sin(LinearIndepedentFunction):
    def call(self, n):
        return np.sin(n)

    def derivative(self):
        def d(n):
            return np.cos(n)
        return d


class Softmax(Function):
    def call(self, n):
        en = np.exp(n)
        return en/np.sum(en)

    def derivative(self):
        '''
        http://eli.thegreenplace.net/2016/
            the-softmax-function-and-its-derivative/

        https://stackoverflow.com/questions/36279904/
            softmax-derivative-in-numpy-approaches-0-implementation

        https://stackoverflow.com/questions/40575841/
        numpy-calculate-the-derivative-of-the-softmax-function
        '''

        def d(n):
            SM = n.reshape((-1, 1))
            jac = np.diag(n) - np.dot(SM, SM.T)

            return jac

        return d

    def jacobian(self):
        return self.derivative()


class Fibonacci(Function):

    def _fib(self, na):
        n = na[0]
        i, j = 1, 0
        for k in range(1, n + 1):
            i, j = j, i + j
        return np.array([j])

    def call(self, n):
        return self._fib(n)

    def derivative(self):
        raise NotImplementedError


class Gaussian(LinearIndepedentFunction):

    def call(self, n):
        return np.exp(-np.power(n, 2))

    def derivative(self):
        raise NotImplementedError
