import numpy
import unittest

from .config import MIN_RATING, MAX_RATING
from .ratings import (
    GaussianRatingsScaler,
    RatingsFactory,
    StandardRatingsScaler
)
from .syntheticmodel import (
    ClustersProductRatingsSampler,
    DecreasingImportanceClustersLayersRatingsSampler,
    EmbeddingsRatingsSampler,
    PureClustersRatingsSampler
)


class InteractionsSamplerTestCase(unittest.TestCase):

    def test_embeddings(self):
        nu, ni = 50, 100
        d = 5
        factory = EmbeddingsRatingsSampler(d).sample(nu, ni)
        assert factory.n_users == nu
        assert factory.n_items == ni
        users = numpy.random.choice(nu, size=100)
        items = numpy.random.choice(ni, size=100)
        ratings = factory.get_ratings(users, items)
        one = 1 + 1e-4
        assert ((ratings >= - one) & (ratings <= one)).all()

    def test_pure_clusters(self):
        nu, ni = 1000, 500
        users = numpy.random.choice(nu, size=10**4)
        items = numpy.random.choice(ni, size=10**4)

        factory = PureClustersRatingsSampler(3).sample(nu, ni)
        ratings = factory.get_ratings(users, items)
        counts = numpy.bincount(ratings.astype(int))
        assert counts.size == 2
        assert numpy.allclose(counts[0], counts[1]*2, rtol=0.1)  # P(same cluster) = 1/3

    def test_clusters_product(self):
        nu, ni = 1000, 500
        users = numpy.random.choice(nu, size=10**4)
        items = numpy.random.choice(ni, size=10**4)

        factory = ClustersProductRatingsSampler((2, 3), unbalanced_factor=1.).sample(nu, ni)
        ratings = factory.get_ratings(users, items)
        counts = numpy.bincount(ratings.astype(int))
        assert counts.size == 2
        assert numpy.allclose(counts[0], counts[1]*5, rtol=0.1)  # P(same cluster) = 1/6

        factory = ClustersProductRatingsSampler((2, 3), unbalanced_factor=10.).sample(nu, ni)
        ratings = factory.get_ratings(users, items)
        counts = numpy.bincount(ratings.astype(int))
        assert counts[1] > counts[0]

    def test_hierarchical_clusters(self):
        nu, ni = 1000, 500
        d = 3
        factory = DecreasingImportanceClustersLayersRatingsSampler(
            d, decrease_factor=1/2).sample(nu, ni)
        users = numpy.random.choice(nu, size=10**4)
        items = numpy.random.choice(ni, size=10**4)
        ratings = factory.get_ratings(users, items) * 2**(d-1)
        counts = numpy.bincount(numpy.round(ratings).astype(int))
        # ratings must be uniformly distributed on {0, 1, ..., 2**d - 1}
        assert counts.size == 2**d, counts
        assert numpy.allclose(counts[0], counts, rtol=0.15)

    def test_standard_scaler(self):
        self._test_scaler(StandardRatingsScaler())

    def test_gaussian_scaler(self):
        self._test_scaler(GaussianRatingsScaler())

    @classmethod
    def _test_scaler(cls, scaler):
        n1 = 5_000
        n2 = 10_000
        scaler = StandardRatingsScaler()
        scaler.fit(numpy.random.randn(n1))
        ratings1 = scaler.transform(numpy.random.randn(n1))
        assert ((MIN_RATING <= ratings1) & (ratings1 <= MAX_RATING)).all()
        assert ratings1.size == n1
        expected_mean = (MIN_RATING + MAX_RATING) / 2
        assert numpy.abs(ratings1.mean() - expected_mean) < 0.2
        assert numpy.abs(numpy.median(ratings1) - expected_mean) < 0.3

        scaler.fit(numpy.random.randn(n1)*1.7 + 4.3)
        ratings2 = scaler.transform(numpy.random.randn(n2)*1.7 + 4.3)
        assert ratings2.size == n2
        test_quantiles = numpy.linspace(0, 1, num=10)
        quantiles1 = numpy.quantile(ratings1, test_quantiles)
        quantiles2 = numpy.quantile(ratings2, test_quantiles)
        assert numpy.allclose(quantiles1, quantiles2, rtol=0, atol=0.4)

        ratings_factory = RatingsFactory(EmbeddingsRatingsSampler(d=4).sample(1000, 1000), scaler)
        users_idx = numpy.random.choice(1000, n1)
        items_idx = numpy.random.choice(1000, n1)
        ratings = ratings_factory.get_ratings(users_idx, items_idx, scale=True)
        assert numpy.abs(ratings.mean() - expected_mean) < 0.2
        assert numpy.abs(numpy.median(ratings) - expected_mean) < 0.3
