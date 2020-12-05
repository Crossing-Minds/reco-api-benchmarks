import numpy
import unittest

from .syntheticdataset import SyntheticDataset


class SyntheticDatasetTestCase(unittest.TestCase):

    def test_sample(self):
        dataset = SyntheticDataset.sample(
            n_users=2_000,
            n_items=2_000,
            synthetic_model='pure-clusters',
            dimension=3,
            interactions_distribution='uniform',
            n_ratings=200_000,
        )

        dataset = SyntheticDataset.sample(
            n_users=2_000,
            n_items=2_000,
            synthetic_model='pure-embeddings',
            interactions_distribution={'user': 'invlog', 'item': 'exponential'},
            ratings_scaling='gaussian',
            interactions_ratings_based='explicit',
            n_ratings=200_000,
            dimension=3,
        )

        dataset = SyntheticDataset.sample(
            n_users=2_000,
            n_items=2_000,
            synthetic_model='clustered-embeddings',
            ratings_scaling='gaussian',
            interactions_distribution='uniform',
            interactions_ratings_based='explicit',
            n_ratings=120_000,
            dimension=4,
        )

    def test_sample_with_features(self):
        dataset = SyntheticDataset.sample(
            n_users=200,
            n_items=200,
            synthetic_model='clusters-product',
            dimension=(2, 3),
            interactions_distribution='uniform',
            n_ratings=2000,
            users_features=['tags1', ('tags2', 0.5)],
            items_features=[('cats1', 0.7), 'cats2']
        )
        assert dataset.users is not None
        assert dataset.items is not None

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

        dataset = SyntheticDataset.sample(
            n_users=200,
            n_items=300,
            synthetic_model='decreasing-clusters-layers',
            dimension=3,
            interactions_distribution='uniform',
            n_ratings=2000,
            users_features=['cats1', None, 'cats1'],
            items_features=['tags2', 'cats2', 'tags1']
        )
        assert len(dataset.items_m2ms)

    def test_get_ratings(self):
        dataset = self._basic_dataset()
        rvals = dataset.get_ratings(dataset.ratings['user_id'], dataset.ratings['item_id'])
        assert numpy.allclose(rvals, dataset.ratings['rating'])

    def test_random_seed(self):
        dataset1= SyntheticDataset.sample(
            n_users=1_000, n_items=1_000, n_ratings=20_000, 
            synthetic_model=('decreasing-clusters-layers', {'decrease_factor': 0.8}),
            dimension=3, interactions_distribution='invlog', 
            interactions_ratings_based='explicit',
            users_features=['tags1', 'cats2'], 
            items_features=['cats1', None, 'tags2'],
            )
        dataset2 = SyntheticDataset.sample(**dataset1.get_config())
        self._assert_equals(dataset1, dataset2)

    def _basic_dataset(self):
        return SyntheticDataset.sample(n_users=1_000, n_items=1_000,
            synthetic_model='pure-clusters', dimension=3,
            interactions_distribution='invlog', n_ratings=20_000)

    def _assert_equals(self, dt1, dt2):
        assert (dt1.ratings == dt2.ratings).all()
        assert (dt1.users == dt2.users).all()
        assert (dt1.items == dt2.items).all()
        for m2ms1, m2ms2 in [
            (dt1.users_m2ms, dt2.users_m2ms),
            (dt1.items_m2ms, dt2.items_m2ms)
        ]:
            for m2m1, m2m2 in zip(m2ms1, m2ms2):
                assert m2m1['name'] ==  m2m2['name']
                assert (m2m1['array'] == m2m2['array']).all()
