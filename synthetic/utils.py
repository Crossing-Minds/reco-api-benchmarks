import random

import numpy
try:
    from numba import njit
except ImportError:
    def njit(x): return x

from .config import MIN_RATING, MAX_RATING


@njit
def set_numba_random_seed(seed):
    """
    :param int seed:

    numba doesn't share the same seed as numpy, so the seed 
    has to be set in a @njit function.
    """
    numpy.random.seed(seed)


def set_random_seed(seed=None):
    """
    :param int? seed:

    set the random for numpy, numba, and native random module.

    :return: int seed
    """
    seed = numpy.random.randint(0, 1 << 30) if seed is None else seed
    random.seed(seed)
    numpy.random.seed(seed)
    set_numba_random_seed(seed)
    return seed


def round_ratings(ratings_val, as_ints=False):
    """
    :param float-array ratings_val: 
    :param bool? as_ints: whether to cast the values as integers or not

    Return the ratings rounded to {MIN_RATING, MIN_RATING + 1, ..., MAX_RATING}
    """
    grid = numpy.linspace(MIN_RATING, MAX_RATING, num=int(MAX_RATING-MIN_RATING+2))
    ratings_val = numpy.clip(numpy.searchsorted(grid, ratings_val), MIN_RATING, MAX_RATING)
    if as_ints:
        ratings_val = ratings_val.astype(int)
    return ratings_val


class DistributionTransport:
    """
    This class allows to transform samples from one origin distribution D
    to map them to another target distribution D' (so it looks like they were sampled for D').
    Samples are (approximately) optimaly transported, 
        c.f. https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)
    This means the order of samples is preserved.
    More precisely, the order is preserved up to `eps`:
    x_i <= x_j + eps => y_i <= y_j where x are the samples from D and y the transformed samples.
    """

    def __init__(self, eps=1e-4, num=200):
        """
        :param float esp: amplitude of the perturbation applied on the samples from D 
                          for enforcing discrete samples to be continuous 
                          (it's required for the implemented optimal transport approximation)
                          If the origin distribution D is continuous, you can pick eps=0. 
        :param int num: number of quantiles used 
                        The higher the more precise the transport.
                        Though it will become more sensitive to the randomness of 
                        samples from the target ditribution D'.
                        Default value is 100, which is a good trade-off.
        """
        self.eps = eps
        self.num = num

    def _to_continuous(self, x):
        """
        :param 1d-array x:
        """
        return x.astype('f8') + numpy.linspace(0, self.eps, num=x.size)

    def fit(self, x, y):
        """
        :param 1d-array x: samples from the origin distribution D (x_i ~iid D)
        :param 1d-array y: samples from the target distribution D' (y_i ~iid D')

        This method computes and stores a mapping from D to D'.
        """
        x = self._to_continuous(x)
        q = numpy.linspace(0, 1, num=self.num)

        self.qx = numpy.quantile(x, q)
        self.qy = numpy.quantile(y, q)
        return self

    def transform(self, x):
        """
        :param 1d-array x: (n,) samples from the origin distribution D (x_i ~iid D)

        :return 1d-array y: (n,) `x` transported to target distribution D'
        """
        x = self._to_continuous(x)
        x = numpy.clip(x, self.qx[0], self.qx[-1])
        qi = numpy.searchsorted(self.qx, x)
        y = self.qy[qi]
        return y


def partition_int(n, q):
    """
    :param int n: int to be partionned
    :param int q: number of partions

    :return int-array p: such that:
        - sum(p) == n
        - len(p) == q
        - for all i, j: |p[i] - p[j]| <= 1
    """
    return n // q + (numpy.arange(q) < n % q)
