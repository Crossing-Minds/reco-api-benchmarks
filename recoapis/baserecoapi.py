import numpy
import time
from functools import wraps

from xminds.compat import logger
from xminds.lib.arrays import set_or_add_to_structured

from synthetic.utils import set_random_seed


def timeit(f):
    """Decorator to time a method. Returns two outputs: the time to execute `f`, and f's outputs"""
    @wraps(f)
    def wrap(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        duration = time.time() - t0
        return duration, result
    return wrap


class BaseRecoApi(object):

    def __init__(self, name, dataset, dataset_hash='', db='test', token='none',
                 environment=str, client=None, algorithm=str,
                 transform_to_implicit=bool, transform_algorithm=str):
        self.name = name
        self.dataset = dataset
        self.dataset_hash = dataset_hash
        self.algorithm = algorithm
        self.transform_to_implicit = transform_to_implicit
        self.transform_algorithm = transform_algorithm
        self.environment = environment  # only used in XMindsAPI
        self.db = db
        self.token = token
        self.client = client

    @timeit
    def timed_reset(self):
        """
        Timed reset
        :returns: float
        """
        self.reset()

    @timeit
    def timed_upload(self):
        """
        Timed upload
        :returns: float
        """
        self.upload()

    @timeit
    def timed_fit(self):
        """
        Timed fit
        :returns: float
        """
        self.fit()

    @timeit
    def timed_recommend(self, *args, **kwargs):
        """
        Timed testing
        :returns: float
        """
        return self.recommend(*args, **kwargs)

    def reset(self):
        """
        If an API can be used by multiple processes (because our pipeline uses separate datasets),
        it is safe to reset after `evaluate`. Otherwise we should reset before `fit`.
        """
        raise NotImplementedError('API-specific')

    def upload(self):
        """
        Prepares data and upload the dataset through the client
        """
        raise NotImplementedError('API-specific')

    def fit(self):
        """
        Train on server and deploy
        :returns: None
        """
        raise NotImplementedError('API-specific')

    def recommend(self, test_user_ids, n_items_per_user=int, exclude_rated_items=bool,
                  reco_delay=float):
        """
        Get recommendations for provided user IDs.
        :param numpy.array test_user_ids: [n,ddtype=uint64] IDs of users to get recos from
        :param int n_items_per_user: number of items recommended per user
        :param bool exclude_rated_items: get recos of items not interacted with in training
        :param float reco_delay: (default 0. normally) time to sleep between two consecutive recos.
            Can set to positive values to reduce server stress
        :returns: (np.array(int32|uint64), np.array(int64), np.array(int64)) (users, items, ranking)
        """
        raise NotImplementedError('API-specific')

    def evaluate(self, users_id, items_id, rankings):
        """
        Calculates various metrics of use:
            - mean_rating (in [1, 10]) is the mean rating of the recommended items
            - proportion_reco_from_training (in [0,1]) calculates ratio of recommended items
            that belong to the training set. Should be 0 in test; may vary later according to use.
        Takes as input the output of `self.recommend` or the idx version (int32)
        if the API uses IDs (uint64)
        :param numpy.array users_idx:
        :param numpy.array items_idx:
        :param numpy.array rankings:
        :returns:  {metric_name: float}
        """
        if users_id == [] and items_id == [] and rankings == []:
            logger.info('Evaluate: users, items, ranking are empty')
            return {}
        users = users_id.astype('int64')
        items = items_id.astype('int64')
        # check which items are not valid (some API had this issue and recommended items with 0
        # rating, possibly from a previous run despite the resetting)
        is_valid = (0 < users) & (users <= self.dataset.n_users) & (0 < items) & (
                items <= self.dataset.n_items)
        valid_users = users[is_valid]
        valid_items = items[is_valid]
        metrics = {'proportion_valid_items': sum(is_valid)/len(is_valid)}

        if hasattr(self.dataset, 'get_ratings'):
            # get average of reco ratings
            mean_rating = self.dataset.get_ratings(valid_users, valid_items).mean()
            metrics.update({'mean_rating': mean_rating})

        # Check that recommended pairs user/item are not in training set
        training_user_item_tuples = {
            (int(u), int(i)) for u, i in zip(
                self.dataset.ratings['user_id'], self.dataset.ratings['item_id'])}
        reco_user_item_tuples = {(int(u), int(i)) for u, i in zip(valid_users, valid_items)}
        reco_in_training = reco_user_item_tuples & training_user_item_tuples
        try:
            prop_rated_items = len(reco_in_training) / len(reco_user_item_tuples)
        except ZeroDivisionError:
            prop_rated_items = numpy.nan
        if prop_rated_items > 0:
            logger.warning(
                f'{100 * prop_rated_items}% of recommendations belong to the training dataset')
        metrics.update({'proportion_reco_from_training': prop_rated_items})

        logger.info(f'Metrics: {metrics}')
        return metrics

    @staticmethod
    def get_test_users(dataset, n_recos):
        """
        Picks users to get recos for.
        Uses dataset's random seed for reproducibility from dataset size.
        n_reco reduced to user range if necessary
        :param SyntheticDataset dataset:
        :param int n_recos: number of users to get recos from. Reduced to
            `dataset.ratings['user'].max() + 1` if need be.
        :returns: [int] users_id
        :raises: AssertionError
        """
        _range = dataset.n_users
        assert _range >= n_recos, f'{n_recos} recos asked but dataset has {_range} users'
        set_random_seed(dataset.config['seed'])
        test_idx = numpy.random.choice(_range, n_recos, replace=False)  # no repetition
        test_ids = dataset._idx_to_id(test_idx).tolist()
        return test_ids

    @staticmethod
    def get_user_item_flat_properties(dataset):
        """
        Generates user and item features according to the type required.
        Examples: 'scalar', ('scalar', 0.3), 'cats1', ...
        item_properties = numpy.unique(self.dataset.ratings['item'])
        may not suffice if API needs at least 1 feature
        :returns: (user_properties, item_properties),
            each a numpy.structarray(['user/item_id', 'scalar_0', ...])
        """
        return dataset.users, dataset.items

    @staticmethod
    def get_user_item_m2m_properties(dataset, asdict=False):
        """
        Generates user and item m2m features according to the type required.
        Examples: 'scalar', ('scalar', 0.3), 'cats1', ...
        :param bool? asdict: (default False) outputs dicts instead of lists
        :returns: (users_m2m_properties, items_m2m_properties),
            each being either, depending on asdict:
             - a list [{'name': str, 'array': structarray([('user/item_index', '<i8'),
                    ('f0-tags', '<u4')]}]
             - a dict {prop_name: array([('user/item_index'), ('value_id')])}
        """
        users_m2ms = dataset.users_m2ms
        items_m2ms = dataset.items_m2ms
        if asdict:
            users_m2ms = {m2m['name']: m2m['array'] for m2m in users_m2ms}
            items_m2ms = {m2m['name']: m2m['array'] for m2m in items_m2ms}
        return users_m2ms, items_m2ms

    # === ratings operation ===

    @staticmethod
    def ratings_to_timestamps(ratings, algorithm):
        """
        Transforms ratings into timestamps
        :param np.array ratings: np.array([n, 'rating'])
        :param str algorithm: in 'random', 'time-linear', 'time'quadratic'
        :returns: np.array([n, 'int64'])
        :raises: NotImplementedError
        """
        now = int(time.time())
        algos = {
            'random': lambda r: now - numpy.random.randint(0, 20000000, len(r)),
            'time-linear': lambda r: now - ((10 - r['rating']) * 24 * 3600),
            'time-quadratic': lambda r: now - (((10 - r['rating']) ** 2) * 24 * 3600)
        }
        try:
            return algos[algorithm](ratings)
        except KeyError:
            raise NotImplementedError(f'Unknown algo {algorithm}')

    @classmethod
    def explicit_to_implicit(cls, dataset, algorithm='linear'):
        """
        Transform explicit ratings into implicit ratings.
        Algorithms:
        - linear: Creates as many interactions with an item as the item is rated
        - quadratic: Creates more interactions than `linear`: the square of the rating value
        - time-linear:  The rating information is put into the timestamp: linearly spread from now,
                        the lowest the rating becoming the oldest timestamp
                        (previously assessed as best for AWS)
        - time-quadratic: Same as `time-linear`, but spreading even more in time
        :param SyntheticDataset dataset: only dataset.ratings is used
        :param str algorithm:  Name of the algorithm to user
        :returns: array([('user',uint32),('item',uint32),('event_type',<U6),('timestamp',uint32)])
        """
        threshold = 5.5  # only keep high ratings
        ratings = dataset.ratings
        new_dt = numpy.dtype([
            ('user_id', numpy.uint32),
            ('item_id', numpy.uint32),
            ('event_type', '<U6'),
            ('timestamp', numpy.uint32)
        ])
        logger.info(f'Transforming explicit into implicit feedback with {algorithm} algorithm')
        if algorithm == 'linear':
            implicit = numpy.repeat(ratings, ratings['rating'].astype(numpy.uint32))
            new_ratings = numpy.empty(implicit.shape, dtype=new_dt)
            users_items = implicit[['user_id', 'item_id']]
            timestamps = time.time()
            event_type = 'rating'
        elif algorithm == 'quadratic':
            implicit = numpy.repeat(ratings, ratings['rating'].astype(numpy.uint32) ** 2)
            new_ratings = numpy.empty(implicit.shape, dtype=new_dt)
            users_items = implicit[['user_id', 'item_id']]
            timestamps = time.time()
            event_type = 'rating'
        elif algorithm == 'time-linear':
            ratings = ratings[ratings['rating'] > threshold]
            timestamps = cls.ratings_to_timestamps(ratings, algorithm)
            new_ratings = numpy.empty(ratings.shape, dtype=new_dt)
            users_items = ratings[['user_id', 'item_id']]
            event_type = 'click'
        elif algorithm == 'time-quadratic':
            ratings = ratings[ratings['rating'] > threshold]
            timestamps = cls.ratings_to_timestamps(ratings, algorithm)
            new_ratings = numpy.empty(ratings.shape, dtype=new_dt)
            users_items = ratings[['user_id', 'item_id']]
            event_type = 'click'
        else:
            raise NotImplementedError(algorithm)
        new_ratings[['user_id', 'item_id']] = users_items
        new_ratings['event_type'] = event_type
        new_ratings['timestamp'] = timestamps
        return new_ratings

    @staticmethod
    def prop_to_kind(prop):
        return numpy.dtype(prop['value_type']).kind

    @classmethod
    def iget_m2m_features(cls, features, features_m2m, astype=None):
        """
        Yield flat and m2m properties (following index of `features`)
        :param struct_array features:  an output of get_user_item_flat_properties
        :param dict features_m2m: an output of get_user_item_m2m_properties(asdict=True)
        :param str? astype: if set, values returned are converted to this type
            (ex: 'U64' to get strings, to use '|'.join() on). `tolist()` applied right after.
        :yield: tuple(flat_features), tuple(list(values)) (ordered as features_m2m.dtype.names)
        """
        m2m_names = features_m2m.keys()
        index = 'user_index' if 'user_id' in features.dtype.names else 'item_index'
        for idx in range(len(features)):
            flat = features[idx]
            m2m = []
            for name in m2m_names:
                mask = features_m2m[name][index] == idx
                if astype:
                    value = features_m2m[name]['value_id'][mask].astype(astype).tolist()
                else:
                    value = features_m2m[name]['value_id'][mask].tolist()
                m2m.append(value)
            yield tuple(flat), tuple(m2m)

    @classmethod
    def get_all_features(cls, features, features_m2m, separator='|'):
        """
        Create struct array regrouping flat and m2m features (as '|'-separated strings)
        :param struct_array features:  an output of get_user_item_flat_properties
        :param dict features_m2m: an output of get_user_item_m2m_properties(asdict=True)
        :param str? separator: (default: '|')
        :returns: structarray(features+features_m2m as strings (separated by separator))
        """
        m2m_names = features_m2m.keys()
        n = numpy.array(
            [tuple(separator.join(kk) for kk in k) for _, k in
             cls.iget_m2m_features(features, features_m2m, astype='U64')],
            dtype=[(name, 'U64') for name in m2m_names])
        all_features = set_or_add_to_structured(
            features, [(name, n[name]) for name in m2m_names])
        return all_features

    def iget_all_features_as_dict(self, features, features_m2m):
        """
        :yield: {'user/item_id': int, flat_prop: scalar, m2m_prop: list}
        """
        for flat_feat, m2m_feat in self.iget_m2m_features(features, features_m2m):
            values = {}
            for i, name in enumerate(features.dtype.names):
                values[name] = flat_feat[i].item()
            for i, name in enumerate(features_m2m):
                values[name] = m2m_feat[i]
            yield values


class RecommendationException(Exception):
    """
    Exception raised for errors expected during recommendation
    :param str expression:  input expression in which the error occurred
    :param str message: explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class TrainingException(Exception):
    """
    Exception raised for errors expected during training
    :param str expression:  input expression in which the error occurred
    :param str message: explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class JobFailedException(Exception):
    """"
    Exception raised for errors in the input.
    :expression: input expression in which the error occurred
    :message: explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class S3Exception(Exception):
    """
    Exception raised for errors in the input.
    :expression: input expression in which the error occurred
    :message: explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
