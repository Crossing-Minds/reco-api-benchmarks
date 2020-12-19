import numpy
import re
import time
from tqdm import tqdm
import random

from recombee_api_client.api_client import RecombeeClient
from recombee_api_client.api_requests import (
    AddDetailView, RecommendItemsToUser, Batch, AddRating, ResetDatabase, AddUserProperty,
    AddItemProperty, SetItemValues)
from recombee_api_client.exceptions import ResponseException
from xminds.compat import logger
from xminds.lib.utils import retry
from xminds.ds.scaling import linearscaling

from .baserecoapi import BaseRecoApi
from .config import RECOMBEE_DBS_TOKENS


class RecombeeRecoApi(BaseRecoApi):

    BATCH_SIZE = 10000
    # Sometime a db might get stuck. In this case switch to another one.

    # List of 2-tuples('API identifier', 'Private token')
    DBS_TOKENS = RECOMBEE_DBS_TOKENS
    DB_TOKEN = None  # is randomised in reset to avoid bugs

    def __init__(self, name, dataset, dataset_hash=None, db=None,
                 token=None, environment=None,
                 algorithm='recombee:personal',
                 transform_to_implicit=False,
                 transform_algorithm=''):
        super().__init__(name, dataset, algorithm=algorithm, db=db, token=token,
                         dataset_hash=dataset_hash, environment=environment,
                         transform_to_implicit=transform_to_implicit,
                         transform_algorithm=transform_algorithm)
        assert self.DBS_TOKENS, 'No ID found. Set varenvs RECOMBEE_API_DB_ID0/TOKEN0 at least'
        self.DB_TOKEN = random.choice(self.DBS_TOKENS)
        self.db = db or self.DB_TOKEN[0]
        self.token = token or self.DB_TOKEN[1]   # private token
        if self.db is None or self.token is None:
            raise RuntimeError(('To use recombee api, db and private token need be provided, '
                                'from credentials or varenv'))
        logger.info('db: %s', self.db)
        logger.info('token: %s', self.token)
        self.client = RecombeeClient(self.db, self.token)

    def get_client(self):
        return self.client

    def upload(self):
        """The dataset is trained after each batch, so the code is left in `self.fit`"""
        pass

    def fit(self):
        logger.info('fit starting. Have usually to wait for the DB to be reset')

        def kind_to_type(prop):
            kind = self.prop_to_kind(prop)
            if kind in 'iu':
                return 'int'
            if kind == 'f':
                return 'double'
            if kind == 'U':
                return 'string'
            raise NotImplementedError(prop)
        dataset = self.dataset
        # items
        users, items = self.get_user_item_flat_properties(dataset)
        users_m2ms, items_m2ms = self.get_user_item_m2m_properties(dataset, asdict=True)
        # create_items_request = [AddItem(item_id) for item_id in items['item_id'].tolist()]
        # resp = send_batch(create_items_request)
        for prop in dataset.iget_items_properties(yield_id=True):
            _type = kind_to_type(prop)
            AddItemProperty(prop['property_name'], _type)
        items_request = []
        for values in self.iget_all_features_as_dict(items, items_m2ms):
            item_id = values.pop('item_id')
            items_request.append(SetItemValues(item_id, values, cascade_create=True))
        self._send_batch(items_request)
        # users
        for prop in dataset.iget_users_properties(yield_id=True):
            _type = kind_to_type(prop)
            AddUserProperty(prop['property_name'], _type)
        users_request = []
        for values in self.iget_all_features_as_dict(users, users_m2ms):
            user_id = values.pop('user_id')
            users_request.append(SetItemValues(user_id, values, cascade_create=True))
        self._send_batch(users_request)
        # ratings preprocessing
        ratings = dataset.ratings
        ratings['rating'] = linearscaling(ratings['rating'], -1, 1)
        #  ratings upload
        ratings_requests = []
        logger.info('transform to implicit: %s', self.transform_to_implicit)
        if self.transform_to_implicit is False:
            for rating in ratings:
                ratings_requests.append(AddRating(str(rating['user_id']), str(rating['item_id']),
                                                  int(rating['rating']), cascade_create=True))
        else:
            logger.info('transform algorithm: %s', self.transform_algorithm)
            new_ratings = self.explicit_to_implicit(dataset, self.transform_algorithm)
            for rating in new_ratings:
                ratings_requests.append(AddDetailView(
                    str(rating['user_id']), str(rating['item_id']),
                    timestamp=str(rating['timestamp']),
                    cascade_create=True))
        batch_size = self.BATCH_SIZE
        n_batches = int(len(ratings_requests) / batch_size)
        extra_batch = len(ratings_requests) % batch_size > 0
        for i in tqdm(range(n_batches)):
            self._send_batch(ratings_requests[i*batch_size:(i+1)*batch_size])
        if extra_batch:
            self._send_batch(ratings_requests[n_batches*batch_size:len(ratings_requests)])

    def recommend(self, test_user_ids, n_items_per_user=32, exclude_rated_items=True, reco_delay=0):
        reco_users = []
        reco_items = []
        reco_data = []
        missing_recos = []
        for i in tqdm(test_user_ids):
            reco = self._get_user_topk_recombee(user_id=i, n_results=n_items_per_user,
                                                exclude_rated_items=exclude_rated_items)
            recomms = reco['recomms']
            if len(recomms) == 0:
                missing_recos.append(i)
                continue
            reco_users.extend([i] * len(recomms))
            reco_items.extend([int(d['id']) for d in recomms])
            reco_data.extend((len(recomms) - numpy.arange(len(recomms))).tolist())
            time.sleep(reco_delay)
        if missing_recos:
            logger.warning(f'{len(missing_recos)} empty recos. First 10: {missing_recos[:10]}')
        result = numpy.array(reco_users), numpy.array(reco_items), numpy.array(reco_data)
        return result

    @retry(base=10, multiplier=1.2, max_retry=2)
    def reset(self):
        """
        Given that deleting is slow and there is no IS_READY endpoint, we test that the DB is
        ready by sending a rating corresponding to a missing item/user
        """
        logger.info(f'Resetting into db {self.db}')
        try:
            self.client.send(ResetDatabase())  # breaks if already being reset
            logger.info('Reset query sent. Sleep 10...')
            time.sleep(10)
        except TypeError as e:
            logger.warning(f'Resetting Recombee dataset failed: e={e}')
            pass
        # wait until the status changes when getting a reco (from 'being erased' to 'missing user')
        user_id = n = 1
        t0 = time.time()
        while True:
            try:
                self.client.send(RecommendItemsToUser(user_id, n, logic={"name": self.algorithm}))
            except ResponseException as e:
                match = re.match(r'.*status:\s*(\d+).*', str(e))
                if not match:
                    raise RuntimeError(f'No status found in error message {e}')
                status = match.group(1)
                if status == '404':  # missing user; the DB has been reset
                    logger.info('DB reset')
                    break
                elif status == '422':  # DB being erased
                    if time.time() - t0 > 3000:
                        raise RuntimeError(f'Resetting did not seem to work: error {e}')
                    logger.info('Not ready yet. Sleep 10')
                    time.sleep(10)
                else:
                    raise NotImplementedError(f'Unknown status {status} from {e}')

    @retry(base=10, multiplier=1.1, max_retry=5)
    def _send_batch(self, batch):
        if not batch:
            # nothing to send
            return
        self.client.send(Batch(batch))

    @retry(max_retry=2)
    def _get_user_topk_recombee(self, user_id, n_results=32, exclude_rated_items=True):
        if not exclude_rated_items:
            filter_ = None
        else:
            mask = self.dataset.ratings['user_id'] == user_id
            items_id = self.dataset.ratings['item_id'][mask].astype('U64').tolist()
            filter_ = "'itemId' not in {" + ",".join([f'"{i}"' for i in items_id]) + '}'
        return self.client.send(RecommendItemsToUser(
            user_id, n_results,
            filter=filter_,
            logic={"name": self.algorithm})
        )
