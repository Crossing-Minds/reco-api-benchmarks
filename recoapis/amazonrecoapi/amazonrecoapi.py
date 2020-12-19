import time
import numpy
from tqdm import tqdm

from botocore.exceptions import ClientError
from xminds.compat import logger
from xminds.lib.utils import deep_hash, retry

from ..baserecoapi import BaseRecoApi
from .clients import AmazonApiClients
from .reset_resources import aws_reset
from .upload import aws_prepare_resources
from .train import aws_train_resources


class AwsArnSet():
    """Contains the attributes necessary for a full run (training/testing)"""
    def __init__(self, users_schema_arn=None, items_schema_arn=None,
                 interactions_schema_arn=None, dataset_group_arn=None, dataset_arn=None,
                 dataset_import_job_arn=None, solution_arn=None, solution_version_arn=None,
                 campaign_arn=None, filter_arn=None):
        self.users_schema_arn = users_schema_arn
        self.items_schema_arn = items_schema_arn
        self.interactions_schema_arn = interactions_schema_arn
        self.dataset_group_arn = dataset_group_arn
        self.dataset_arn = dataset_arn
        self.dataset_import_job_arn = dataset_import_job_arn
        self.solution_arn = solution_arn
        self.solution_version_arn = solution_version_arn
        self.campaign_arn = campaign_arn
        self.filter_arn = filter_arn


class AmazonRecoApi(BaseRecoApi):

    def __init__(self, name, dataset, dataset_hash='', algorithm='hrnn',
                 users_schema_arn=None, items_schema_arn=None,
                 interactions_schema_arn=None, dataset_group_arn=None, dataset_arn=None,
                 dataset_import_job_arn=None, solution_arn=None, solution_version_arn=None,
                 campaign_arn=None, filter_arn=None,
                 environment=None,
                 transform_to_implicit=True,
                 transform_algorithm='time-linear'):
        """
        For transform_to_implicit: On trivial dataset False gave random score.
            True gave bad score but better than random
        For algorithm: 'hrnn' is of recipe type USER_PERSONALIZATION, 'sims' of type RELATED_ITEMS
        """
        assert algorithm in ('hrnn', 'hrnn-meta', 'sims', 'user-personalization'), (
            f'{algorithm} not allowed')
        super().__init__(name, dataset,
                         dataset_hash=dataset_hash, algorithm=algorithm,
                         environment=environment, transform_to_implicit=transform_to_implicit,
                         transform_algorithm=transform_algorithm)
        self.arn_set = AwsArnSet(
            users_schema_arn=users_schema_arn,
            items_schema_arn = items_schema_arn,
            interactions_schema_arn = interactions_schema_arn,
            dataset_group_arn = dataset_group_arn,
            dataset_arn = dataset_arn,
            dataset_import_job_arn = dataset_import_job_arn,
            solution_arn = solution_arn,
            solution_version_arn = solution_version_arn,
            campaign_arn = campaign_arn,
            filter_arn = filter_arn,
        )
        self.api_clients = AmazonApiClients()
        self.event_type = ('click' if self.transform_to_implicit else 'rating')
        fit_params = {  # used to get a hash for this run
            'algo': self.algorithm,
            'etype': self.event_type,
            '2impl': self.transform_to_implicit,
            'transf_algo': self.transform_algorithm
        }
        _api_hash = deep_hash(fit_params, fmt='hex40')
        logger.info(f'Amazon: _api_hash={_api_hash} from {fit_params}')
        short_hash = f'{self.name}_{self.dataset_hash[:10]}_{_api_hash[:10]}'
        self.resources_name =  short_hash # short hash
        assert len(short_hash) < 45, f'String `{short_hash}` Should be under 45 (total len<63)'

    @retry(max_retry=10, multiplier=1.3, base=10)
    def reset(self, only_self=True):
        """
        Method to reset the data (campaigns, solutions, dataset_groups) in AWS's bucket.
        Might need to be called many times, slowly, to wait for each step to be deleted before
        the next one is (ex: can only delete a solution when the campaigns referring to it are
        deleted
        :param bool only_self: (Default True) If False, clean the whole
            bucket. If True, only the attributes of `self` are reset (self.solution_arn, ...)
        """
        aws_reset(self.arn_set, only_self, self.api_clients.personalize_client)

    def upload(self):
        try:
            self._upload()
        except (ClientError, NotImplementedError) as e:
            logger.error('Exception in aws endpoint: %s. All resources created will be deleted', e)
            self.reset(only_self=True)
            raise

    def fit(self):
        """
        During model training, Amazon Personalize considers a maximum of 750 thousand items.
        If you import more than 750 thousand items, Amazon Personalize decides which items to
        include in training, with an emphasis on including new items (items you recently created
        with no interactions) and existing items with recent interactions data.
        event_type: 'click' or 'rating' depending on transf2implicit
        """
        try:
            self._fit()
        except (ClientError, NotImplementedError) as e:
            logger.error('Exception in aws endpoint: %s. All resources created will be deleted', e)
            self.reset(only_self=True)
            raise

    def recommend(self, test_user_ids, n_items_per_user=32, exclude_rated_items=True, reco_delay=0):
        """
        The filter could be created here, but given that there is a little delay until it is
        activated, it is creating in the beginning, even if not used for reco.
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/personalize.html#Personalize.Client.create_filter
        """
        reco_users = []
        reco_items = []
        reco_data = []
        self._wait_filter_active()
        for i in tqdm(test_user_ids):
            reco = self._get_user_topk_amazon(
                self.arn_set.campaign_arn, user_id=i, n_results=n_items_per_user,
                exclude_rated_items=exclude_rated_items,
                filter_arn=self.arn_set.filter_arn
            )
            reco_users.extend([i] * len(reco))
            reco_items.extend([int(d['itemId']) for d in reco])
            reco_data.extend((len(reco) - numpy.arange(len(reco))).tolist())
            time.sleep(reco_delay)
        result = numpy.array(reco_users), numpy.array(reco_items), numpy.array(reco_data)
        return result

    def _upload(self):
        users, items = self.get_user_item_flat_properties(self.dataset)
        users_m2ms, items_m2ms = self.get_user_item_m2m_properties(self.dataset, asdict=True)
        all_user_features = self.get_all_features(users, users_m2ms)
        all_item_features = self.get_all_features(items, items_m2ms)
        aws_prepare_resources(self.api_clients, self.arn_set,
                              all_user_features, all_item_features, self.dataset,
                              self.resources_name, self.transform_to_implicit,
                              self.transform_algorithm)

    def _wait_filter_active(self):
        t0 = time.time()
        while True:
            if time.time() - t0 > 200:
                raise RuntimeError(f'Filter {self.arn_set.filter_arn} still not active. Issue?')
            resp = self.api_clients.personalize_client.describe_filter(
                filterArn=self.arn_set.filter_arn)
            status = resp['filter']['status']
            if status == 'ACTIVE':
                logger.info('Filter active')
                return
            logger.info(f'Filter {self.arn_set.filter_arn} not active. Sleep(3)...')
            time.sleep(3)

    def _fit(self):
        aws_train_resources(
            self.api_clients, self.arn_set, self.algorithm, self.event_type, self.resources_name)

    def _get_user_topk_amazon(
            self, campaign_arn, user_id, n_results=30, exclude_rated_items=True, filter_arn=None):
        """
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/personalize-runtime.html#PersonalizeRuntime.Client.get_recommendations
        """
        if exclude_rated_items:
            get_recommendations_response = self.api_clients.runtime_client.get_recommendations(
                campaignArn=campaign_arn,
                userId=str(user_id),
                numResults=n_results,
                filterArn=filter_arn,
            )
        else:
            get_recommendations_response = self.api_clients.runtime_client.get_recommendations(
                campaignArn=campaign_arn,
                userId=str(user_id),
                numResults=n_results,
            )
        item_list = get_recommendations_response['itemList']
        return item_list

