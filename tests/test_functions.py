import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from cbrain.function import LMS, Square, Fibonacci, CrossEntropy
from cbrain.function import Line, Der, ReLU, Softmax, Jac, Gaussian


class TestFunctions(unittest.TestCase):

    def test_LMS(self):
        l = LMS()
        t = np.array([5, 7])
        o = np.array([7, 11])

        assert_array_equal(l(t, o), np.array([20]))
        assert_array_equal(Der(l)(t, o), np.array([4, 8]))
        assert_array_equal(Jac(l)(t, o), np.array([[4, 0], [0, 8]]))

    def test_CrossEntropy(self):
        c = CrossEntropy()
        t = np.array([5, 7])
        o = np.array([7, 11])

        call_ret = np.array(-26.5148176549)
        der_ret = np.array([-0.71428571, -0.63636364])
        jac_ret = np.array([[-0.71428571,  0], [0, -0.63636364]])

        assert_almost_equal(c(t, o), call_ret)
        assert_almost_equal(Der(c)(t, o), der_ret)
        assert_almost_equal(Jac(c)(t, o), jac_ret)

    def test_Square(self):
        s = Square()
        i = np.array([5, 12])

        assert_array_equal(s(i), np.array([25, 144]))
        assert_array_equal(Der(s)(i), np.array([10, 24]))
        assert_array_equal(Jac(s)(i), np.array([[10, 0], [0, 24]]))

    def test_Fibonacci(self):
        f = Fibonacci()
        i = np.array([7])

        self.assertEqual(f(i), np.array([13]))

    def test_Line(self):
        f = Line(2, 4)
        i = np.array([1, 2, 3, 4])

        assert_array_equal(f(i), np.array([6, 8, 10, 12]))
        assert_array_equal(Der(f)(i), np.array([2, 2, 2, 2]))
        assert_array_equal(Jac(f)(i), np.diag(np.array([2, 2, 2, 2])))

    def test_ReLU(self):
        f = ReLU()
        i = np.array([1, -2, 3, -4])

        assert_array_equal(f(i), np.array([1, 0, 3, 0]))
        assert_array_equal(Der(f)(i), np.array([1, 0, 1, 0]))
        assert_array_equal(Jac(f)(i), np.diag(np.array([1, 0, 1, 0])))

    def test_Softmax(self):
        f = Softmax()
        i = np.array([1, 2, 3])

        f_array = np.array([0.09003057, 0.24472847, 0.66524096])
        assert_almost_equal(f(i), f_array)

        der_array = np.array([[0, -2, -3], [-2, -2, -6], [-3, -6, -6]])
        assert_array_equal(Der(f)(i), der_array)
        assert_array_equal(Jac(f)(i), der_array)

    def test_Gaussian(self):
        f = Gaussian()
        i = np.array([1, 2, 3])

        f_array = np.array([3.67879441e-01, 1.83156389e-02, 1.23409804e-04])
        assert_almost_equal(f(i), f_array)



if __name__ == "__main__":
    unittest.main()
