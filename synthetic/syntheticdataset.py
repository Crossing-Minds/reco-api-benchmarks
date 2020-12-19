import numpy
import json

from xminds.lib.arrays import set_or_add_to_structured, to_structured

from .config import EXPLICIT_RATINGS_DISTRIBUTION, IMPLICIT_RATINGS_DISTRIBUTION
from .features import (
    BaseFeatureSampler, CategoriesSampler, ScalarSampler, TagsSampler,
    sample_features
)
from .interactionssampler import InteractionsSampler
from .ratings import GaussianRatingsScaler, RatingsFactory, StandardRatingsScaler
from .syntheticmodel import (
    ClusteredEmbeddingsRatingsSampler,
    ClustersLayersRatingsSampler,
    ClustersProductRatingsSampler,
    DecreasingImportanceClustersLayersRatingsSampler,
    DecreasingImportanceEmbeddingsRatingsSampler,
    EmbeddingsRatingsSampler, 
    PureClustersRatingsSampler, 
    RandomRatingsSampler
)
from .utils import set_random_seed


class SyntheticDataset:
    ID_DTYPE = 'uint32'

    def __init__(self, ratings, users, items, users_m2ms, items_m2ms, ratings_factory, config):
        """
        Private constructor. Please use `SyntheticDataset.sample` instead
        :param RATINGS_DTYPE-array ratings:
        :param struct-array users: users ids and eventual users features 
        :param struct-array items: items ids and eventual items features
        :param list users_m2ms: eventual m2m users features 
        :param list items_m2ms: eventual m2m items features 
        :param RatingsFactory ratings_factory:
        :param dict config:
        """
        self.n_users = users.size
        self.n_items = items.size
        self.ratings = ratings
        self.users = users
        self.items = items
        self.users_m2ms = users_m2ms
        self.items_m2ms = items_m2ms
        self.ratings_factory = ratings_factory
        self.config = config

    @classmethod
    def sample(
        cls,
        n_users=1000,
        n_items=1000,
        n_ratings=10_000,
        synthetic_model='pure-clusters',
        dimension=4,
        ratings_scaling='standard',
        interactions_distribution='uniform',
        interactions_ratings_based=False,
        users_features=[],
        items_features=[],
        seed=None
    ):
        """
        :param int? n_users: 
        :param int? n_items:
        :param int? n_ratings: 
        :param string-or-tuple? synthetic_model: Type of synthetic model to be used.
        :param int-or-tuple? dimension: Describes the dimension of the synthetic model. 
        :param str? ratings_scaling: How ratings are scaled into [1, 10]. Either 'standard'
            or 'gaussian'.
        :param str-or-tuple? interactions_distribution: How interactions are distributed among
            users and items.
        :param str-or-False? interactions_ratings_based: Whether and how to bias interactions
            sampling.
        :param list? users_features: Describes the users features. 
        :param list? items_features: Same as `users_features` but for items.

        Possible values for `synthetic_model`:
        - 'pure-embeddings'
        - 'decreasing-pure-embeddings' | additionnal kwarg(s): decrease_factor=0.9
        - 'clustered-embeddings' | additionnal kwarg(s): n_clusters=None, ortho_fraction=1.,
            cluster_scale=4., normalize=True
        - 'pure-clusters' 
        - 'clusters-product' | additionnal kwarg(s): decrease_factor=0.9, unbalanced_factor=1.
        - 'clusters-layers' | additionnal kwarg(s): unbalanced_factor=None
        - 'decreasing-clusters-layers': 
        See the docstring of the corresponding classes (in `ratingssampler.py`) for details
            about each possibility
        
        Possible values for `dimension`:
        - the embeddings dimension for 'pure-embeddings', 'decreasing-pure-embeddings'
            or 'clustered-embeddings'
        - the number of clusters for 'pure-clusters'
        - the number of clusters-layers for 'clusters-product', 
            if dimension is a int-tuple, each int is the number of clusters for the
                corresponding layer (so the number of layers is the length of the tuple)

        Meaning of the possible values of `ratings_scaling`:
        - 'standard': scale ratings in [1, 10] preserving the shape of the ratings distribution
        - 'gaussian': scale ratings in [1, 10] transporting the shape of the ratings to a
            truncated gaussian distibution centered in 5.5

        Possible values for `interactions_distribution`:
        - 'uniform': the number of interactions of each user/item will roughly be the same.
        - 'exponential': the number of interactions of users/items follow an
            exponential distribution
        - 'invlog': the number of interactions of users/items is distributed even more unevenly
            than for 'exponential'
        - a dict {'user': distribution, 'item': distribution}
        - tuple(one of above, addition_kwargs). See the docstring of `InteractionSampler`
            for details.

        Possible values for `interactions_ratings_based`:
        - False: interactions will be missing at random 
        - 'explicit': models MNAR for datasets such as MovieLens
        - 'implicit': models MNAR for implicit feedback datasets 
        More information about MNAR can be found in the `InteractionSampler` docstring.
        
        About features:
        Users/items features are based on one dimension/layer of the users/items synthetic truth.
        This synthetic truth can either be embeddings, clusterings, layers of clusterings, ...
        The length of the given list must be at most the number of dimension/layer of the
            synthetic truth.
        Warning: for 'pure-clusters', this number is 1. Try 'clusters-product' for
            fully-clustered datasets but with higher number of synthetic truth layers.
        Possible values for elements of the `users_features` or `items_features` list:
        - a string: the name of the feature. Either:
            - 'scalar': scalar values
            - 'cats1': simple categories
            - 'cats2': more complex categories
            - 'tags1': simple tags
            - 'tags2': more complex tags
        - a tuple (string, float): the name and the difficulty of the feature
            the difficulty (in [0, 1]) defines how hard it is to get valuable information from
                this feature:
                - 0 means straight-forward / very easy (default is 0)
                - 1 means impossible
        - None: no feature for this dimension.
        - a subclass of BaseFeatureSampler
        """
        assert n_ratings / n_users >= 1, 'less than 1 item per user, please increase `n_ratings`'
        assert n_ratings / n_items >= 1, 'less than 1 user per item, please increase `n_ratings`'

        seed = set_random_seed(seed)
        config = dict(
            n_users=n_users,
            n_items=n_items,
            n_ratings=n_ratings,
            synthetic_model=synthetic_model,
            dimension=dimension,
            ratings_scaling=ratings_scaling,
            interactions_distribution=interactions_distribution,
            interactions_ratings_based=interactions_ratings_based,
            users_features=users_features,
            items_features=items_features,
            seed=seed
        )

        # 1) Ratings sampler
        sm_kwargs = {}
        if type(synthetic_model) is tuple:
            synthetic_model, sm_kwargs = synthetic_model

        if synthetic_model == 'pure-embeddings':
            ratings_sampler = EmbeddingsRatingsSampler(dimension, **sm_kwargs)
        elif synthetic_model == 'decreasing-pure-embeddings':
            ratings_sampler = DecreasingImportanceEmbeddingsRatingsSampler(
                dimension, **sm_kwargs)
        elif synthetic_model == 'clustered-embeddings':
            ratings_sampler = ClusteredEmbeddingsRatingsSampler(dimension, **sm_kwargs)
        elif synthetic_model == 'pure-clusters':
            ratings_sampler = PureClustersRatingsSampler(dimension, **sm_kwargs)
        elif synthetic_model == 'clusters-product':
            ratings_sampler = ClustersProductRatingsSampler(dimension, **sm_kwargs)
        elif synthetic_model == 'clusters-layers':
            ratings_sampler = ClustersLayersRatingsSampler(dimension, **sm_kwargs)
        elif synthetic_model == 'decreasing-clusters-layers':
            ratings_sampler = DecreasingImportanceClustersLayersRatingsSampler(
                dimension, **sm_kwargs)
        elif synthetic_model == 'random':
            ratings_sampler = RandomRatingsSampler()
        else:
            raise ValueError(f'Unknown synthetic model: {synthetic_model}')

        # 2) Ratings scaler
        if ratings_scaling == 'standard':
            ratings_scaler = StandardRatingsScaler()
        elif ratings_scaling == 'gaussian':
            msg = f'{synthetic_model} is not compatible with gaussian ratings scaling.'
            assert synthetic_model not in ['pure-clusters', 'clusters-product'], msg
            ratings_scaler = GaussianRatingsScaler()
        else:
            raise ValueError(f'Unknown ratings scaling method: {ratings_scaling}')

        # 3) Sample users&items truth and instanciate the ratings factory
        synthetic_model = ratings_sampler.sample(n_users, n_items)
        ratings_factory = RatingsFactory(synthetic_model, ratings_scaler)

        # 4) Sample the interactions :
        interaction_sampler_kwargs = {}
        if isinstance(interactions_distribution, (tuple, list)):
            interactions_distribution, interaction_sampler_kwargs = interactions_distribution
            interaction_sampler_kwargs = {**interaction_sampler_kwargs}
        if type(interactions_distribution) is str:
            interactions_distribution = {
                'user': interactions_distribution,
                'item': interactions_distribution
            }
        interaction_sampler_kwargs['users_distribution'] = interactions_distribution['user']
        interaction_sampler_kwargs['items_distribution'] = interactions_distribution['item']
        if interactions_ratings_based == 'explicit':
            interaction_sampler_kwargs['target_ratings_distribution'] = EXPLICIT_RATINGS_DISTRIBUTION
        elif interactions_ratings_based == 'implicit':
            interaction_sampler_kwargs['target_ratings_distribution'] = IMPLICIT_RATINGS_DISTRIBUTION
        density = n_ratings / (n_items * n_users)
        interactions_sampler = InteractionsSampler(density, **interaction_sampler_kwargs)
        interactions = interactions_sampler.sample(n_users, n_items, ratings_factory)

        # 5) Users/Items features
        users_features = list(cls._iparse_features(users_features))
        items_features = list(cls._iparse_features(items_features))
        users, users_m2ms = sample_features(users_features, 'user', synthetic_model.users_truth)
        items, items_m2ms = sample_features(items_features, 'item', synthetic_model.items_truth)

        # 6) Transform it to the right format
        ratings_val = ratings_factory.get_ratings(interactions['user'], interactions['item'])
        ratings = to_structured([
            ('user_id', cls._idx_to_id(interactions['user'])),
            ('item_id', cls._idx_to_id(interactions['item'])),
            ('rating', ratings_val)
        ])
        users_id = [('user_id', cls._idx_to_id(numpy.arange(n_users)))]
        if users is None:
            users = to_structured(users_id)
        else:
            users = set_or_add_to_structured(users, users_id)
        items_id = [('item_id', cls._idx_to_id(numpy.arange(n_items)))]
        if items is None:
            items = to_structured(items_id)
        else:
            items = set_or_add_to_structured(items, items_id)
        
        return cls(ratings, users, items, users_m2ms, items_m2ms, ratings_factory, config)

    def get_ratings(self, users_id, items_id):
        """
        :param (n,)-ID_DTYPE-array users_ids:
        :param (n,)-ID_DTYPE-array items_ids:

        :returns: (n,)-float-array ratings
        """
        users_idx = self._id_to_idx(users_id)
        items_idx = self._id_to_idx(items_id)
        return self.ratings_factory.get_ratings(users_idx, items_idx)

    def __str__(self):
        s = f'Synthetic dataset of {self.n_users} users x {self.n_items} items, '
        s += f'{self.ratings.size} ratings'
        return s

    def iget_items_properties(self, yield_id=False):
        return self._iget_properties(self.items, self.items_m2ms, 'item', yield_id=yield_id)

    def iget_users_properties(self, yield_id=False):
        return self._iget_properties(self.users, self.users_m2ms, 'user', yield_id=yield_id)

    def get_config(self):
        return self.config

    def save(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.get_config(), f)

    @classmethod
    def load(cls, fname):
        with open(fname) as f:
            config = json.load(f)
        return cls.sample(**config)

    @classmethod
    def _id_to_idx(cls, ids):
        return ids - 1

    @classmethod
    def _idx_to_id(cls, idxs):
        return idxs.astype(cls.ID_DTYPE) + 1

    @classmethod
    def _iget_properties(cls, features, m2ms, features_of, yield_id):
        assert features_of in ['user', 'item']
        for col in features.dtype.names:
            if col == f'{features_of}_id' and not yield_id:
                continue
            yield {
                'property_name': col,
                'value_type': str(features[col].dtype),
                'repeated': False
            }
        for m2m in m2ms:
            yield {
                'property_name': m2m['name'],
                'value_type': str(m2m['array']['value_id'].dtype),
                'repeated': True
            }

    @classmethod
    def _iparse_features(cls, features):
        """
        :param list features: 
        Parse the users/items features list passed to __init__
        See __init__ docstring for more precision on what is allowed in the list. 

        :generates: BaseFeatureSampler
        """
        for feature in features:
            if type(feature) is str:
                yield cls._feature_name_to_feature(feature)
            elif type(feature) in (tuple, list) and len(feature) == 2:
                name, difficulty = feature
                yield cls._feature_name_to_feature(name, difficulty)
            elif issubclass(type(feature), BaseFeatureSampler):
                yield feature
            elif feature is None:
                yield None
            else:
                raise TypeError(f'Invalid feature: {feature}')

    @classmethod
    def _feature_name_to_feature(cls, name, difficulty=0):
        """
        :param string name: name of the feature 
        :param float? difficulty: in [0, 1]

        :returns: BaseFeatureSampler
        """
        msg = f'Difficulty of feature {name} is not in [0, 1] ({difficulty}).'
        assert 0 <= difficulty <= 1, msg
        if name == 'tags1':
            return TagsSampler(difficulty=difficulty, clusters_tag_dim=20,
                               n_tags_range=(2, 5), unbalanced_factor=0)
        elif name == 'tags2':
            return TagsSampler(difficulty=difficulty, clusters_tag_dim=None,
                               n_tags_range=(1, 7), unbalanced_factor=2)
        elif name == 'cats1':
            return CategoriesSampler(dims=1, difficulty=difficulty)
        elif name == 'cats2':
            dims = 3 + numpy.arange(1000) % 3
            return CategoriesSampler(dims=dims, difficulty=difficulty)
        elif name == 'scalar':
            return ScalarSampler(difficulty=difficulty)
        else:
            raise NameError(f'Invalid feature name: {name}.')
