import time
import numpy
from tqdm import tqdm

from apiclients.client import ApiClientInternal
from xminds.compat import logger
from xminds.lib.utils import retry
from xminds.lib.arrays import to_structured

from .baserecoapi import BaseRecoApi
from .config import XMINDS_API_USER, XMINDS_API_PASSWORD


class XMindsRecoApi(BaseRecoApi):

    TEST_NAME = 'Test API Benchmark Synthetic DB'
    API_USER = XMINDS_API_USER
    API_PASSWORD = XMINDS_API_PASSWORD

    def __init__(self, name, dataset, dataset_hash=None, db='test', token='none',
                 environment='staging', algorithm='default', transform_to_implicit=False,
                 transform_algorithm=''):
        super().__init__(name, dataset,
                         db=db, token=token, dataset_hash=dataset_hash, environment=environment,
                         algorithm=algorithm, transform_algorithm=transform_algorithm,
                         transform_to_implicit=transform_to_implicit)
        assert algorithm in ('default',), f'Only `default` algorithm usable on prod for the moment'
        logger.info(f'XMinds user:{self.API_USER}, env:{environment}, pwd:{self.API_PASSWORD[:3]}.')
        self.db_name = f'{self.TEST_NAME}_{self.dataset_hash}'
        self.client = None   # requires _init_client
        assert self.API_USER, 'varenv XMINDS_API_B2B_BENCHMARK_USER must be set'
        assert self.API_PASSWORD, 'varenv XMINDS_API_B2B_BENCHMARK_PASSWORD must be set'

    def _init_client(self):
        """Before `fit` we need to login"""
        if self.client is not None:
            return
        self.client = ApiClientInternal()
        self.client.login_root(self.API_USER, self.API_PASSWORD)

    @retry()
    def _create_db(self, id_type='uint64'):
        """
        uint32 too small for the user IDs
        """
        assert self.client is not None
        db = self.client.create_database(
            name=self.db_name,
            description='Testing Crossing Minds API Experiment on Synthethic Data',
            item_id_type=id_type,
            user_id_type=id_type
        )
        logger.info(f'Creating db {db["id"]}')
        self.client.login_individual(self.API_USER, self.API_PASSWORD, db['id'])
        self.db = db

    @retry(max_retry=5, multiplier=1.15, base=10)
    def _reset_databases(self, client, only_self=False):
        """
        Depending on `only_self`, delete all test DB or only the current one
        Using `retry` delete may have some timeout issues
        Using `retry` is fine here (despite a POST) because there is uniqueness of DB name
        """
        dbs = client.get_all_databases()
        for db in dbs['databases']:
            case1 = (not only_self) and db['name'].startswith(self.TEST_NAME)
            case2 = only_self and db['name'] == self.db_name
            if case1 or case2:
                logger.info(f'XMinds API: delete db {db}')
                client.login_individual(self.API_USER, self.API_PASSWORD, db['id'])
                client.delete_database()

    @retry(max_retry=5, multiplier=1.15, base=10)
    def _delete_database(self):
        """After getting recos, we can delete the current db we are logged in"""
        self.client.delete_database()

    def upload(self):
        """
        Upload data through client
        Beware of ID types; if int/uint, ID `0` is not accepted so
        need to use IDs (uint64 necessary for user IDs)
        """
        self._init_client()
        self._reset_databases(self.client, only_self=True)
        # removes db of same name if any; shouldn't happen but can
        self._create_db()
        # Prepare ratings
        items_id = self.dataset.ratings['item_id']
        users_id = self.dataset.ratings['user_id']
        ratings = to_structured([
            ('user_id', users_id),
            ('item_id', items_id),
            ('rating', self.dataset.ratings['rating']),
            ('timestamp', time.time())],
        )
        # items
        users, items = self.get_user_item_flat_properties(self.dataset)
        users_m2ms, items_m2ms = self.get_user_item_m2m_properties(self.dataset)
        item_properties = list(self.dataset.iget_items_properties())
        for p in item_properties:
            self.client.create_item_property(**p)
        logger.info(f'Uploading items bulk...')
        self.client.create_or_update_items_bulk(items, items_m2ms, chunk_size=1000)
        # users
        user_properties = list(self.dataset.iget_users_properties())
        if user_properties:
            for p in user_properties:
                self.client.create_user_property(**p)
            logger.info(f'Uploading users bulk...')
            self.client.create_or_update_users_bulk(users, users_m2ms, chunk_size=1000)
        # ratings
        logger.info(f'Updating ratings bulk...')
        self.client.create_or_update_ratings_bulk(ratings, chunk_size=(1<<13))

    def reset(self, only_self=True):
        self._init_client()
        self._reset_databases(self.client, only_self=only_self)

    @retry()  # only safe to retry because we reset the db
    def fit(self,):
        @retry(max_retry=5, multiplier=1.5, base=5)  # may be started slightly too soon
        def trigger_and_wait():
            # Long timeout for simplicity
            self.client.trigger_and_wait_background_task(
                'ml_model_retrain', timeout=1234567, lock_wait_timeout=150000, verbose=True)

        trigger_and_wait()

    def recommend(self, test_user_ids, n_items_per_user=32,
                  exclude_rated_items=True, reco_delay=0):
        """
        :param test_user_ids: test's users IDs
        :param int? n_items_per_user: (default: 32) max number of items to recommend per reco
        :param float? reco_delay:  (default 0.0) sleep after getting a reco to reduce server stress
        :param bool? exclude_rated_items: (default True) excludes already rated items
        :returns:  numpy.array(reco_users), numpy.array(reco_items), numpy.array(reco_data)
        """
        reco_users = []
        reco_items = []
        reco_data = []
        missing_recos = []

        @retry(max_retry=4, multiplier=1.25, base=2)  # possible errors due to timing
        def get_reco(_id, amt):
            return self.client.get_reco_user_to_items(
                _id, amt=amt, exclude_rated_items=exclude_rated_items)

        logger.info(f'Getting {len(test_user_ids)} recos')
        for _id in tqdm(test_user_ids):
            resp = get_reco(_id, n_items_per_user)
            time.sleep(reco_delay)
            items_id = resp['items_id']
            if len(items_id) == 0:
                missing_recos.append(_id)
                continue
            reco_users.extend([_id] * len(items_id))
            reco_items.extend(items_id)
            reco_data.extend((len(items_id) - numpy.arange(len(items_id))).tolist())
        if missing_recos:
            logger.warning(f'"{len(missing_recos)}" IDs brought no recommendation, '
                           f'including {missing_recos[:10]}')
        return numpy.array(reco_users), numpy.array(reco_items), numpy.array(reco_data)
