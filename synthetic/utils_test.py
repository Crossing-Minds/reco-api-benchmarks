import random

import numpy
from scipy.stats import rankdata
import unittest

from .utils import set_random_seed, DistributionTransport, njit, partition_int


class InteractionsSamplerTestCase(unittest.TestCase):

    def test_set_random_seed(self):
        n = 1 << 30
        seed = set_random_seed()
        rand1 = random.randint(0, n)
        np_rand1 = numpy.random.randint(n)
        nb_rand1 = self._numba_randint(n)
        set_random_seed(seed)
        rand2 = random.randint(0, n)
        np_rand2 = numpy.random.randint(n)
        nb_rand2 = self._numba_randint(n)
        assert rand1 == rand2
        assert np_rand1 == np_rand2
        assert nb_rand1 == nb_rand2

    def test_distribution_transport(self):
        n = 10_000
        x = numpy.random.rand(n)  # x_i ~iid D = U[0, 1]
        xp = numpy.random.rand(2 * n)  # xp_i ~iid D = U[0, 1]
        y = 5 + numpy.random.randn(3 * n) + numpy.random.randint(0, 2, size=3 * n)
        # y_i ~idd D' = 5 + N(0, 1) + B(1/2)

        yp = DistributionTransport(eps=0).fit(x, y).transform(xp)
        # => yp looks like samples form D'

        def distribs_distance(x, y):
            q = numpy.linspace(0.01, 0.99, num=300)
            qx = numpy.quantile(x, q)
            qy = numpy.quantile(y, q)
            return numpy.abs(qx - qy).sum()

        # check that yp looks much closer to D' than xp
        assert distribs_distance(yp, y) < (distribs_distance(xp, y) / 50)
        # check that the order is completly preserved (it should be as `eps=0`)
        rank_xp = rankdata(xp)
        rmin = rankdata(yp, method='min')
        rmax = rankdata(yp, method='max')
        assert (rmin <= rank_xp).all()
        assert (rmax >= rank_xp).all()

    def test_partition_int(self):
        for n in range(100):
            for q in range(1, 50):
                p = partition_int(n, q)
                assert sum(p) == n
                assert len(p) == q
                assert p.max() - p.min() <= 1

    @staticmethod
    @njit
    def _numba_randint(a):
        return numpy.random.randint(a)
