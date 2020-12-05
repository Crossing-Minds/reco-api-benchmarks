import numpy
import unittest

from xminds.lib.arrays import unique_count, kargmax

from .interactionssampler import InteractionsSampler
from .syntheticdataset import SyntheticDataset


class InteractionsSamplerTestCase(unittest.TestCase):

    def test_uniform(self):
        density = 0.01
        n_users = 500
        n_items = 1000

        sampler = InteractionsSampler(density)
        ratings = sampler.sample(n_users, n_items)

        # check we have approximately the expected number of interactions
        n_ratings_expected = density * n_users * n_items
        assert 0.95 < ratings.size / n_ratings_expected < 1.05

        # test no duplicated_interactions
        self._test_no_duplicated_interactions(ratings)

        # most users should have around 10 interactions
        n_ratings_per_user = numpy.bincount(ratings['user'])
        nrpu_repartion = numpy.bincount(n_ratings_per_user) / n_users
        assert nrpu_repartion[7:15].sum() > 0.7

        # most items should have around 5 interactions
        n_ratings_per_item = numpy.bincount(ratings['item'])
        nrpi_repartion = numpy.bincount(n_ratings_per_item) / n_items
        assert nrpi_repartion[3:8].sum().sum() > 0.7

    def test_ensure_one(self):
        density = 1 / 1000
        n_users = 1000
        n_items = 1000

        sampler_ensure_one = InteractionsSampler(density, ensure_one_per_item=True)
        ratings = sampler_ensure_one.sample(n_users, n_items)
        # ensure_one is True for users and items
        assert numpy.all(numpy.bincount(ratings['user']) >= numpy.ones(n_users, dtype=int))
        assert numpy.all(numpy.bincount(ratings['item']) >= numpy.ones(n_items, dtype=int))

    def test_exponential(self):
        density = 0.02
        n_users = 1000
        n_items = 2000

        exp_sampler = InteractionsSampler(density, users_distribution='invlog', items_distribution='exponential',
                                          min_per_user=2, ensure_one_per_item=False)
        exp_ratings = exp_sampler.sample(n_users, n_items)
        # check we have approximately the expected number of interactions
        n_ratings_expected = density * n_users * n_items
        assert 0.95 < exp_ratings.size / n_ratings_expected < 1.05
        assert (numpy.bincount(exp_ratings['user']) >= 2).all()

        # test no duplicated_interactions
        self._test_no_duplicated_interactions(exp_ratings)

        # with exponential sampler, most active users/popular items should
        # represent a larger part of ratings than with uniform sampler
        unif_sampler = InteractionsSampler(density)
        unif_ratings = unif_sampler.sample(n_users, n_items)
        for k in [1, 5, 10, 50, 100]:
            assert (self._proportion_of_top_k(exp_ratings['user'], k) >
                    self._proportion_of_top_k(unif_ratings['user'], k))
            assert (self._proportion_of_top_k(exp_ratings['item'], k) >
                    self._proportion_of_top_k(unif_ratings['item'], k))

    def test_ratings_based(self):
        dataset = SyntheticDataset.sample(
            n_users=200,
            n_items=300,
            synthetic_model='clustered-embeddings',
            interactions_distribution='invlog',
            ratings_scaling='gaussian',
            interactions_ratings_based='explicit',
            n_ratings=2000,
            dimension=3,
            users_features=[None, ('scalar', 0.5)],
            items_features=['scalar']*3
        )

        counts, _ = numpy.histogram(dataset.ratings['rating'], range=(1, 10))
        assert counts[0] < counts[-1], counts
        assert counts[1]*3 < counts[-2], counts

    @classmethod
    def _test_no_duplicated_interactions(cls, ratings):
        assert unique_count(ratings[['user', 'item']]) == ratings.size

    @classmethod
    def _proportion_of_top_k(cls, entities, k):
        counts = numpy.bincount(entities)
        top_k = kargmax(counts, k)
        return counts[top_k].sum() / counts.sum()
